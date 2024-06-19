Overview
========

By now you have hopefully installed the code and downloaded a grid file. You're now ready to start using `synthesizer`.

In this section we briefly describe the main elements of the code, and the design philosophy.

Grids
*****

`Grids`` are one of the fundamental components in synthesizer.
At its simplest, a grid describes the emission as a function of some parameters.
Typically these are the age and metallicity of a stellar population, and the emission is derived from a stellar population synthesis (SPS) model.

However, these parameters can be arbitrary, of any number of dimensions, and the emission can describe any source.
For example, one could have a grid that has been post-processed through a photoionisation code, where the ionisation parameter is changed, or the source could be the emission from the narrow line region of an active galactic nuclei.
Different grids can also be swapped in and out to assess the impact of different modelling choices; for example, one might wish to understand the impact of different SPS models on the integrated stellar emission.

We provide a number of pre-computed grids for stellar and AGN emission that will be sufficient for most use cases (see `here <../grids/grids.rst>`_ for details).
Advanced users can also generate their own grids, using the `synthesizer-grids` package (see `here <../advanced/creating_grids.rst>`_).


Particle vs Parametric
**********************

Synthesizer can be used to generate the multi-wavelength emission from a range of astrophysical models with a wide array of complexity and fidelity.
At one end, simple toy models can be generated within synthesizer that describe a galaxy through analytic forms; at the other end, data from high resolution isolated galaxy simulations can be ingested into synthesizer, consisting of tens of thousands of discrete elements describing the galaxy properties.

Wherever your data source lies on this spectrum of complexity, it can typically be described in one of two 
These data sources can primarily be divided into two types: **Particle** or **Parametric**.

Particle data represents an astrophysical object through discrete elements with individual properties.
These can describe, for example, the spatial distribution of stellar mass, or the ages of individual star elements.

Conversely, Parametric data typically represents a galaxy through *binned attributes*.
This binning can be represented along different dimensions representing various properties of the galaxy.
An example of this is the star formation history; a parametric galaxy would describe this history by dividing the mass formed into bins of age.

Whilst both of these approaches may appear to be superficially similar, there are some important distinctions under the hood within synthesizer.
In most use cases `synthesizer` will be smart enough to know what kind of data you are providing, and create the appropriate objects as required.
However, it is worth understanding this distinction, particularly when debugging any issues.
We provide examples for various tasks in synthesizer using both particle and parametric approaches where applicable.

Galaxies & Components
*********************

The main object within `synthesizer` is a `Galaxy`. A `Galaxy` object describes various integrated properties of a galaxy, as well as containing individual *components*.
These components can include:

* A stellar component
* A gas component
* Black hole components

A component in `synthesizer`` can be initialised and used independently of a `Galaxy` object, and the emission from that individual component can also be generated.
However, much of the power of synthesizer comes from combining these components; a `Galaxy` object simplifies how they interact with one another, making the self-consistent generation of complex spectra from various components within a galaxy simpler and faster.

Emission models
***************

The generation of spectra in `synthesizer` is handled through *Emission Models*. 
An emission model is a set of standardised procedures for generating spectra from a `Galaxy` or a component.
These often take one of four forms: extraction, combinations, generation, and attenuation.
Further details are provided in the 
`Emission Models <../emission_models/emission_model.ipynb>`_ section.
