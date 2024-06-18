Overview
========

By now you have hopefully installed the code and downloaded a grid file. You're now ready to start using `synthesizer`.

In this section we briefly describe the design philosophy and the main elements of the code.

Particle vs Parametric
**********************

Synthesizer can be used to generate astrophysical emission from a number of different data sources.
These data sources can primarily be divided into two types: **Particle** or **Parametric**.

Particle data describes an astrophysical object through discrete elements with individual properties.
These can describe, for example, the stellar mass spatial distribution, or the ages of individual star elements.

Conversely, Parametric data typically describes a galaxy through attributes binned along different dimensions.
An example of this is the star formation history; a parametric galaxy would describe this history by dividing the mass formed into bins of age.

Whilst both of these approaches may appear to be superficially similar, there are some important distinctions under the hood within synthesizer.
In most use cases `synthesizer` will be smart enough to know what kind of data you are providing, and create the appropriate objects as required.
However, it is worth understanding this distinction, particularly when trying to debug any issues.

Galaxies & Components
*********************

The main base object within `synthesizer` is a `Galaxy` object. This describes various integrated properties of a galaxy, as well as individual *components*.
These can include:

* A stellar component
* A gas component
* Black hole components

Each component within synthesizer can be initialised and used independently of a `Galaxy` object, and used to predict the emission from it.
However, much of the power of synthesizer comes from combining these components; a `Galaxy` object simplifies how they interact with one another, making the generation of complex spectra from various components simpler and faster.

Emission models
***************

The generation of spectra in `synthesizer` is handled through *Emission Models*. 
An emission model is a set of standardised procedures for generating spectra from a `Galaxy` or a component.
These often take one of four forms: extraction, combinations, generation, and attenuation.
Further details are provided in the 
`Emission Models <../emission_models/emission_model.ipynb>`_ section.
