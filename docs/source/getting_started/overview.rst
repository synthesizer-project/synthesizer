Overview
========

Synthesizer is a C-accelerated Python package for generating synthetic observations from theoretical models. In this section you will find a brief overview of the code, its philosophy, and the main tools that it uses.

Philosophy
~~~~~~~~~~

Synthesizer is intended to be modular, flexible and fast.

To enable this, Synthesizer is designed around a simple workflow into which the user can plug in their own models, data, and parameters. This workflow takes theoretical **inputs** (i.e. a stellar population synthesis library, an AGN model, etc.) and theoretical **emitters** (i.e. particle distributions from a cosmological simulation, or parametric models), combines them with an **emission model** that defines the method for translating the inputs into an emission, and then (optionally) applies **instrument** properties to convert theoretical emission into synthetic **observables** (i.e. photometry, spectra, images, etc.). 
Each of these steps is encapsulated in an object, which can be swapped out, specialised, or extended to suit the user's needs. 

While this workflow was designed with this framework in mind, each of the tools can be used independently to fit into whatever workflow or use case the user has in mind. It's also worth pointing out at this point that the framework's flexibility means it can be used for a number of tasks, not necessarily limited to generating observables from cosmological simulations.

It is also worth noting that Synthesizer is not intended as a replacement for detailed codes for generating synthetic galaxy emission that leverage radiative transfer techniques (e.g. `SKIRT <https://skirt.ugent.be/root/_home.html>`_, `Powderday <https://powderday.readthedocs.io/en/latest/>`_).
Instead, Synthesizer is intended to be much cheaper computationally, allowing an exploration of parameter and model dependencies.

Particle vs Parametric
**********************

Synthesizer can be used to generate multi-wavelength emission from a range of astrophysical models with a wide array of complexity and fidelity.
At one end, simple toy models can be generated within Synthesizer, describing a galaxy through analytic forms; at the other end, data from high resolution isolated galaxy simulations can be ingested into Synthesizer, consisting of tens of thousands of discrete elements sampling the matter of a galaxy. In between these two extremes are a myriad of other ways of describing a galaxy, from semi-analytic models (SAMs) to cosmological simulations with varying degrees of resolution and complexity. Synthesizer's goal is to provide a toolset that will work across this entire spectrum of complexity, allowing users to generate synthetic observations from a wide range of astrophysical models.

Wherever your data source lies on this spectrum of complexity, it can typically be described as belonging to one of two types: **Particle** or **Parametric** data.

**Particle** data represents an astrophysical object through discrete elements with individual properties.
These can describe, for example, the spatial distribution of stellar mass, the extent of the discrete element, or the ages of individual star elements.
We use the term 'particle' here in the most general form to describe a discrete resolution element; whether that's a particle element in a smoothed particle hydrodynamics simulation, or a grid element in an adaptive mesh refinement code.

Conversely, **Parametric** data typically represents a galaxy through *binned attributes*.
This binning can be represented along different dimensions representing various properties of the galaxy.
An example of this is the star formation history; a parametric galaxy would describe this history by dividing the mass formed into bins of age.

Whilst both of these approaches may appear to be superficially similar, there are some important distinctions under the hood within Synthesizer.
In most use cases, Synthesizer will be smart enough to know what kind of data you are providing, and create the appropriate objects and call the appropriate methods itself.
However, it is worth understanding this distinction, particularly when debugging any issues.
We provide examples for various tasks in synthesizer using both particle and parametric approaches where applicable.



The Synthesizer Toolbox
~~~~~~~~~~~~~~~~~~~~~~~

Synthesizer is structured around a set of core abstractions, here we give a brief outline of these abstractions and their purpose to explain the design ethos underpinning Synthesizer.

Emission Grids
**************

``Grids`` are one of the fundamental inputs in Synthesizer. A ``Grid`` object holds an N-dimensional array of spectra and emission lines indexed by some parameters. The exact parameters depend on the type of grid (e.g. grids of emission from stellar populations, AGN line regions, dust emission), but they can effectively be anything. 
For stars, these are typically the age and metallicity of a stellar population, indexing emissions derived with a stellar population synthesis (SPS) model (see `Conroy 2013 <https://arxiv.org/abs/1301.7095>`_ for a review).
Alternatively, a more complex set of axes could include a changing ionisation parameter used in a photoionisation code. 
Different grids can also be swapped in and out to assess the impact of different modelling choices; for example, one might wish to understand the impact of different SPS models on the integrated stellar emission.

Synthesizer provides a suite of `pre-computed grids <../emission_grids/grids.rst>`_ from models including `BC03 <https://ui.adsabs.harvard.edu/abs/2003MNRAS.344.1000B>`_, `BPASS <https://ui.adsabs.harvard.edu/abs/2018MNRAS.479...75S>`_, `FSPS <https://ui.adsabs.harvard.edu/abs/2009ApJ...699..486C>`_, `Maraston <https://ui.adsabs.harvard.edu/abs/2025arXiv250103133N>`_, and a series of AGN models derived from `AGNSED <https://ui.adsabs.harvard.edu/abs/2018MNRAS.480.1247K/abstract>`_, all of which have been reprocessed using Cloudy for a number of different photoionisation prescriptions and axes sets. Users can also generate custom grids via the accompanying `grid-generation package <https://github.com/synthesizer-project/grid-generation>`_ (see `here <../advanced/creating_grids.rst>`_), specifying variations in IMF, ionisation parameter, density, and geometry.


Components & Galaxies
~~~~~~~~~~~~~~~~~~~~~

Components 
**********

Components are containers for your "emitters". They include ``Stars``, ``Gas``, and ``BlackHoles`` objects, which are each used to represent the stellar, gaseous, and black hole components of a galaxy respectively. These Components can be parametric models, Semi-Analytic Model outputs or hydrodynamical simulation outputs.  Each of these objects defines methods for calculating properties (e.g. star formation histories, integrated quantities, bolometric luminosities etc.), setting up a model (e.g. calculating line of sight optical depths, dust screens optical depths, dust to metal ratios etc.), and generating observables (e.g. spectra, emission lines, images, and spectral data cubes), along with a number of helper methods for working with the resulting emissions and observables (e.g. analysing and plotting).

Galaxies 
********

While the user is free to work with components directly, a ``Galaxy`` object can be used to combine components and define galaxy-wide properties such as redshift and galactic centre. Like the components, the Galaxy object provides methods for calculating properties, setting up a model, and generating observables. However, the ``Galaxy`` object also provides methods for utilising multiple components at once for more complex models.

Emission models
***************

At the core of Synthesizer's flexibility and modularity are ``EmissionModel`` objects. These are templates defining every step in the process of translating components into emissions. Each individual ``EmissionModel`` can define one of 4 operations:

- Extraction: Extracting emissions from a ``Grid``.
- Generation: Generating emissions from a parametric model.
- Transformation: Transforming an emission into a new emission.
- Combination: Combining multiple emissions together.

Chaining together these 4 ``EmissionModel`` operations results in a modular network, where each of the individual models can be swapped out for an alternative ``EmissionModel`` (or multiple models).
Further details are provided in the `Emission Models <../emission_models/emission_models.rst>`_ section.

Emissions
*********

Applying an Emission Model to a ``Galaxy`` and its components, yields ``Sed`` objects, holding spectra, or a ``LineCollection`` objects, holding emission lines depending on the method called. These objects provide methods for manipulating, analysing, and visualising their contents, including methods to convert emissions from luminosities to fluxes. For instance, ``Sed`` objects contain a variety of useful methods for accessing the luminosity, flux and wavelength, as well as other more specific properties and derived properties (for example, the strength of the Balmer break), while ``LineCollection`` objects provide methods for accessing the line fluxes, equivalent widths, and combining lines into composite lines (e.g. doublets, triplets, etc.).

Emissions can be converted into observables by applying an ``Instrument`` or ``InstrumentCollection`` object to them.


Observatories & Instruments
***************************

To convert an emission into an observable the properties of an observatory must be applied. This is parametrised by the ``Instrument`` object, a flexible container designed to hold the properties of any type of observatory, including photometric imagers, spectrographs, and IFU instruments.

While many of the properties are simple values (i.e. a resolution or resolving power), certain instruments require more detailed properties. For example, a photometric imager ``Instrument`` needs a description of the filter transmission curves. These are encapsulated by the ``FilterCollection`` object. These filters can be user defined, using an explicit transmission curve or the limits of a top-hat filter. More powerfully, however, Synthesizer provides an interface to the `Spanish Virtual Observatory (SVO) filter database <https://svo2.cab.inta-csic.es/theory/fps/>`_, which allows users to easily use any filter from the database by simply passing a filter name to the ``FilterCollection`` at instantiation.


Observables
***********

By combining an emission object with an ``Instrument`` or ``InstrumentCollection``, Synthesizer can translate the theoretical emission into an observable accounting for observational effects.
Observables include spectra (accounting for resolving power and noise, again in ``Sed`` objects), photometry (``PhotometryCollection`` objects), images (``Image`` and ``ImageCollection`` objects), and spectral data cubes (``SpectralDataCube`` objects). Just like emissions, observables are not just containers, they provide a number of methods for manipulating, analysing, and visualising their contents.


