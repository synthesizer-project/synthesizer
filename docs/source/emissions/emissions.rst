Emissions 
=================

Emissions are the theoretical outputs from combining a ``Galaxy`` and/or its ``Components`` with an ``EmissionModel`` (though they can also be extracted from a `Grid <../emission_grids/grids.rst>`_ or created explicitly with a user input). In this section, we will cover generating and working with these objects.

Emission Objects 
~~~~~~~~~~~~~~~~

There are two containers for emissions in Synthesizer: the ``Sed`` and the ``LineCollection``, for spectra and emission lines respectively. These objects are used to store the results of emission calculations, and provide methods for manipulating and analysing the emissions. A summary of each of these objects can be found in the following sections. 

.. toctree::
   :maxdepth: 1

   emission_objects/sed_example 
   emission_objects/lines_example


Generating Emissions 
~~~~~~~~~~~~~~~~~~~~

Synthesizer currently supports the generation of spectra and lines for both stellar populations and AGN.
Generation of these emissions requires an `emission model <../emission_models/emission_models>`_ defining the different spectra to be generated and how they should be combined and manipulated. 
With a model in hand, all that needs to be done is pass that model to the ``get_spectra`` or ``get_lines`` method on the ``Component`` or ``Galaxy``. 

Integrated vs Particle
----------------------

Synthesizer enables the generation of both integrated emissions (a single spectra/set of lines per component / galaxy) or "per-particle" emissions (a spectra/set of lines for each individual particle).
The latter of these is (unsurprisingly) only applicable for particle components (`particle.Stars` and `particle.BlackHoles`), while the former can be generated for both parametric and particle components.
Galaxy level emissions are always, by definition, integrated.

It is worth noting that integrated emissions are always generated regardless of whether a particular call to ``get_spectra`` or ``get_lines`` is generating per-particle emissions or not. This is because the integrated calculation is sufficiently cheap in terms of memory and time that it is worth doing regardless.

In the examples below, we demonstrate this process for individual components and galaxies. Since there is a lot of degeneracy between the different components (its the same process for each), we only show the process for each component for spectra. We also demonstrate exploring lines from a ``Grid`` in the final example. 

.. toctree::
   :maxdepth: 1

   spectra/stars
   spectra/blackholes
   spectra/galaxy
   lines/galaxy_lines
   lines/grid_lines




