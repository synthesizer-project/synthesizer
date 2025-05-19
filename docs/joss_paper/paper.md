---
title: "Synthesizer: Forward modelling with everything but the kitchen sink"
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
bibliography: paper.bib
---

# Summary

Synthesizer is a fast, flexible, modular, and extensible C accelerated Python package for forward modelling both parametric models and numerical simulation outputs. It provides a unified framework to translate astrophysical components (stars, gas, black holes) into synthetic observables—including spectra, photometry, imaging, and spectral data cubes—using a wide variety of stellar population synthesis models, photoionisation assumptions, and dust prescriptions. To ensure the package is as performant as possible, Synthesizer employs shared memory parallelism with OpenMP in C extensions for all computationally intensive calculations with the scope for the user to leverage hybrid parallelism with MPI in the users ecosystem. Synthesizer enables rapid forward modelling of large simulation catalogs, and exploration of the impact of modelling choices with a simple, well documented, and performant set of tools.

# Statement of need

Comparing theoretical models of galaxy formation with observations traditionally relies on computationally expensive radiative transfer codes (e.g. ...) to translate models inot the observer space, or simplified inverse modelling approaches (e.g., SED fitting ...) to translate observations into the theoretical space. The latter of these approaches introduces numerous biases and uncertainties based not only on observational effects but also model assumptions. Compounding these uncertainties is the fact that converged inverse modelling techniques are costly in their own right meaning they must often simplify down the parameter space they explore to ensure convergence in a reasonable time. Both these problems make the former option of forward modelling from the theoretical space to the observer space an attractive prospect.

However, many existing forward modelling tools lack the flexibility to explore modelling uncertainties, the usability and modularity to explore a wide rage of modelling assumptions, the performance necessary to explore a large parameter space and process modern day large datasets, and in many cases they lack documentation and thus consistency and reproducibility across a range datasets.

Synthesizer addresses these shortcomings by offering:

- Flexibility: Nearly every model component (e.g. input spectra grid, escape fraction, dust attenuation law) can be configured, replaced, or extended without modifying core code. Indeed, entire models can be swapped out with minimal effort.

- Performance: Core operations are optimized by employing C extensions with OpenMP threading to enable fast generation of observables, suitable for large simulation volumes or training datasets for simulation-based inference.

- Modularity: The code is designed from the ground up around building block objects (e.g. Grid, Components, EmissionModel, AttenuationLaw), each

- Extensibility: Users can easily construct their own bespoke models, adding new SPS grids, photoionisation recipes, dust models, and instrument definitions without ever modifying the core code.

Synthesizer's design facilitates apples-to-apples comparisons between simulations and observations, promotes reproducible science, and enables the forward modelling of large datasets previously considered impractical.

# Design Ethos

Synthesizer is structured around 5 core abstractions:

Galaxy and Components:

Grids: Precomputed N-dimensional arrays of spectra and lines indexed by parameters such as age, metallicity, ionisation parameter, or density (all axes are arbitrary).

Emission Models:

Instruments:

Emissions & Observables:

# Citations

# Figures

# Acknowledgements

# References
