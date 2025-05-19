---
title: "Synthesizer: Synthetic Observables For Modern Astronomy"
tags:
  - Python
  - astronomy
  - astrophysics
  - forward modelling
  - simulations
authors:
  - name: Will Roper
    orcid: 0000-0002-3257-8806
    corresponding: true
    equal-contrib: true
    affiliation: 1
  - name: Christopher Lovell
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 2
  - name: Aswin Vijayan
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 1
  - name: Stephen Wilkins
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 1
  - name: Hollis Akins
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 3
  - name: Sabrina Berger
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "4, 5"
  - name: Connor Sant Fournier
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 6
  - name: Thomas Harvey
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 7
  - name: Kartheik Iyer
    orcid: 0000-0001-9298-3523
    equal-contrib: true
    affiliation: 8
  - name: Marco Leonardi
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 9
  - name: Sophie Newman
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 2
  - name: Ashley Perry
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 1
  - name: Louise Seeyave
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 1
  - name: Laura Sommovigo
    orcid: 0000-0002-2906-2200
    equal-contrib: true
    affiliation: 10

affiliations:
  - name: Astronomy Centre, University of Sussex, Falmer, Brighton BN1 9QH, UK
    index: 1
    ror: 00ayhx656
  - name: Institute of Cosmology and Gravitation, University of Portsmouth, Burnaby Road, Portsmouth, PO1 3FX, UK
    index: 2
  - name: Department of Astronomy, The University of Texas at Austin, Austin, TX 78712, USA
    index: 3
  - name: School of Physics, University of Melbourne, Parkville, VIC 3010, Australia
    index: 4
  - name: ARC Centre of Excellence for All Sky Astrophysics in 3 Dimensions (ASTRO 3D). Australia
    index: 5
  - name: Department of Physics, University of Malta, Msida MSD 2080, Malta
    index: 6
  - name: Jodrell Bank Centre for Astrophysics, University of Manchester, Oxford Road, Manchester M13 9PL, UK
    index: 7
  - name: Columbia Astrophysics Laboratory, Columbia University, 550 West 120th Street, New York, NY 10027, USA
    index: 8
  - name: Leiden Observatory, Leiden University, PO Box 9513, NL-2300 RA Leiden, The Netherlands
    index: 9
  - name: Center for Computational Astrophysics, Flatiron Institute, 162 5th Ave, New York, NY 10010, USA
    index: 10

date: 15 May 2025
bibliography: synthesizer.bib
---

# Summary

Synthesizer is a fast, flexible, modular, and extensible C accelerated Python package for forward modelling both parametric models and numerical simulation outputs. It provides a unified framework to translate astrophysical components (stars, gas, black holes) into synthetic observables—including spectra, photometry, imaging, and spectral data cubes—using a wide variety of stellar population synthesis models, photoionisation assumptions, and dust model prescriptions. To ensure the package is as performant as possible, Synthesizer offloads all computationally intensive calculations to C extensions and employs explicit shared memory parallelism with OpenMP in these while leaving the user free to employ hybrid parallelism with MPI themselves. Synthesizer enables rapid forward modelling and exploration of modelling choice impact with a simple, well documented, and performant set of tools.

# Statement of need

Comparing theoretical models of galaxy formation with observations traditionally relies on computationally expensive radiative transfer codes (e.g. ...) to translate models inot the observer space, or simplified inverse modelling approaches (e.g., SED fitting ...) to translate observations into the theoretical space. The latter of these approaches introduces numerous biases and uncertainties based not only on observational effects but also model assumptions. Compounding these uncertainties is the fact that converged inverse modelling techniques are costly in their own right meaning they must often simplify down the parameter space they explore to ensure convergence in a reasonable time. Both these problems make the former option of forward modelling from the theoretical space to the observer space an attractive prospect.

However, many existing forward modelling tools lack the flexibility to explore modelling uncertainties, the usability and modularity to explore a wide rage of modelling assumptions, the performance necessary to explore a large parameter space and process modern day large datasets, and in many cases they lack documentation and thus consistency and reproducibility across a range datasets.

Synthesizer addresses these shortcomings by offering:

- Flexibility: Anything that could potentially be changed by the user is designed from the ground up to either be variable (e.g. escape and covering fractions, dust law parameters, model parameters) or swappable with an alternative (e.g. input SPS/AGN grids of spectra, dust law parametrisation, thermal emission parametrisation). This means that users can easily vary any parts of a model in a reproducible way without needing to modify the core code.

- Performance: Computationally intensive operations are optimized by employing C extensions with OpenMP threading where applicable, to enable fast generation of observables, suitable for large simulation volumes or training datasets for simulation-based inference. Without this performance the aforementioned flexbility is moot, only by coupling flexibility with the performance to utilise it can we explore large parameter spaces in a reasonable time.

- Modularity: Synthesizer is object-orientated, with a focus on decoupled classes that can be specialised and then swapped out at will. This modularity, in conjunction with Synthesizer's reliance on templating (where different modular in Synthesizer are templated together) and dependency injection (where the template is translated into an emission), is what enables Synthesizer's flexibility.

- Extensibility: Each of Synthesizer's modular building blocks are designed with a clear API, enabling users to extend the package with their own calculations, input models, and parametrisations. From the beginning, Synthesizer has been designed to be an ecosystem which can be expanded to fit the needs of all users, even as astronomy and astrophysics evolves.

Synthesizer's design facilitates apples-to-apples comparisons between simulations and observations, exhaustive tests of modelling parameter impact, promotes reproducible science, and enables the forward modelling of large datasets previously considered impractical.

# Design and implmentation

Synthesizer is structured around 6 core abstractions, each with a part to play in the forward modelling process. These abstractions are detailed below.

## Component and Galaxy objects

Containers that hold the particle or parametric data from which emissions and observables will be generated.

These classes are containers for the user's parametric models, Semi-Analytic Model outputs or hydrodynamical simulation outputs, and thus are the main computation element in Synthesizer. A Component can be a Stars, Gas, or BlackHoles object, and a Galaxy is a container for these components. Each of these objects defines a number of methods for calculating properties (e.g. star formation histories, integrated quantities, bolometric luminosities etc.), setting up a model (e.g. calculating line of sight optical depths, dust screens optical depths, dust to metal ratios etc.), and generating observables (e.g. spectra, emission lines, images, and spectral data cubes), along with a number of helper methods for working with the resulting emissions and observables (e.g. analysing and plotting).

## Grid objects

Precomputed N-dimensional arrays of spectra and lines indexed by parameters such as age, metallicity, ionisation parameter, or density (all axes are arbitrary).

Synthesizer provides a suite of precomputed SPS grids from models including BC03 [@bc03], BPASS [@bpass], FSPS [@fsps1, @fsps2]. All of which having been photoionisation-processed grids using Cloudy for a number of different photoionisation prescriptions. Users can also generate custom grids via the accompanying [grid-generation package](https://github.com/synthesizer-project/grid-generation), specifying variations in IMF, ionisation parameter, density, and geometry.

## EmissionModel objects

Templates that define every step in the process of translating Components and a Galaxy into emissions and observables. These templates are a modular network of Emission Models, each of which can be swapped out for an alternative Emission Model (or many models). This is the core of Synthesizer's flexibility and modularity.

## Emission objects

The result of applying an Emission Model to a Galaxy and its Components, these Emissions include Seds which hold spectra and LineCollections which hold emission lines.

## Instrument objects

These objects define the properties of an instrument. When combined with a Galaxy and its Components these can be used to translate emissions into observables from an instrument with a specific set of properties. As with all other objects in Synthesizer, these objects are modular and can be swapped out for an alternative Instrument object. Instruments can be combined together to produce IntrumentCollections, which can be used to produce observables from multiple instruments at once.

An instrument defining a photometric imager will also contain a FilterCollection object, which defines the properties of the filters used by the instrument. These filters can be user defined, using an explicit transmission curve or defining a top hat filter. Importantly, Synthesizer provides an interface to the Spanish Virtual Observatory (SVO) filter database, which also allows users to easily use any filter from the database in an Instrument object or in a standalone FilterCollection object.

## Observables

# Citations

# Figures

# Acknowledgements

# References
