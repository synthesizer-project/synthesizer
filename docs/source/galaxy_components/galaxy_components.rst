Galaxies and Their Components 
=============================

Synthesizer enables the generation of emissions from both whole galaxies and their individual components, such as stars and black holes (gas emission planned for a future version). To enable this, Synthesizer uses a modular approach, where galaxies are represented by the ``Galaxy`` container object, and the individual components are represented by their own objects (e.g. ``Stars``, ``BlackHoles``) attached to the ``Galaxy`` object. 

In this section, we will cover instantiating components and galaxies, and how to work with them.

Particle vs Parametric
~~~~~~~~~~~~~~~~~~~~~~

Synthesizer can generate emissions and observables from simulations and parametric models. These two approaches require substantially different data structures and workflows. Galaxy and component objects therefore take two different forms depending on the data representation: *particle* or *parametric*. However, short of instantiting the right the user has to do very little differently with Synthesizer handling almost all of the differences internally. Indeed, for galaxy instantiation we also provide a factory function which will return the correct type of galaxy object based on the data provided. This is demonstrated in the example below in the galaxy object section.

Components
~~~~~~~~~~

Components are the individual parts of a galaxy that contribute to its emission. These include stars, gas, and black holes. Each component has its own object in Synthesizer, which provides methods for generating emissions and observables from that component, as well as methods for manipulating and analysing the component's properties.

Each of these components can be initialised independent of a ``Galaxy`` object, and each can be coupled with a ``Grid`` object to produce their own emission (which we cover in the `emissions <../emissions/emissions.rst>`_ section). When inistantiating a component, any additional properties can be provided as keyword arguments, which will be stored as attributes of the component object.

In the following pages we describe these components, and how to initialise them in parametric or particle form.

.. toctree::
   :maxdepth: 1

   stars
   gas
   blackholes


The Galaxy Object
~~~~~~~~~~~~~~~~~

A ``Galaxy`` is essentially a container object for different components, and provides methods for interacting with and combining these components.
Importantly, this includes methods for producing emissions and observables from the galaxy as a whole and from the individual components.

Global galaxy properties
^^^^^^^^^^^^^^^^^^^^^^^^

In addition to the component attributes, a galaxy can also hold galaxy level attributes. 
These include a ``name`` for the galaxy, and more importantly the redshift of the galaxy, an attribute required to calculate the observer frame emission of the galaxy, and the spatial centre of the galaxy (a property required for imaging).
Beyond the redshift, and like any other container object in Synthesizer, the user can provide additional kwargs to the galaxy object, which will be stored as galaxy level attributes.
This enables the storing of arbitrary data needed later in a pipeline (e.g. predefined optical depths).

In the examples below we demonstate how to instantiate a galaxy object using a factory function, and how to use a galaxy to compute line of sight column densities by combining its components. For further details on generating emissions and observables from a galaxy see the `emissions <../emissions/emissions.rst>`_ and `observables <../observables/observables.rst>`_ sections of the documentation.

.. toctree::
   :maxdepth: 1

   particle_parametric
   line_of_sight

