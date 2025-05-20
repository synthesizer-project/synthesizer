---
title: "Synthesizer: Synthetic Observables For Modern Astronomy"
tags:
  - Python
  - astronomy
  - astrophysics
  - forward modelling
  - simulations
authors:
  - name: Will J. Roper
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
license: GPL-3.0
bibliography: synthesizer.bib
---

# Summary

Synthesizer is a fast, flexible, modular, and extensible Python package that empowers astronomers to turn theoretical galaxy models into realistic synthetic observations-spectra, photometry, images, and spectral cubes-with a focus on interchangeable modelling assumptions. By offloading computationally intensive tasks to threaded C extensions, Synthesizer delivers both simplicity and speed, enabling rapid forward-modelling workflows without requiring users to manage low-level details.

# Statement of need

Comparing theoretical models of galaxy formation with observations traditionally relies on two main approaches. The first uses computationally expensive radiative transfer codes—where fidelity is prioritized over performance (e.g. @sunrise, @SKIRT, @powderday)—to translate models into observer space (forward modelling). The second uses simpler, bespoke pipelines that sacrifice some physical fidelity to generate observables rapidly from large datasets (e.g. @Fortuni2023, @Marshall2025).

Simplified inverse modelling approaches, such as SED fitting (e.g. @EAZY; @BAGPIPES; @PROSPECTOR), work in the opposite direction by translating observations into the theoretical space. However, these methods introduce biases and uncertainties from both observational effects and model assumptions. Compounding these uncertainties is the fact that converged inverse modelling techniques are costly in their own right, necessitating a simplified parameter space to ensure convergence in a reasonable time. Forward modelling is therefore becoming increasingly important not only for probing the validity of theoretical models, but also for quantifying the uncertainties in the modelling assumptions themselves.

However, existing forward modelling tools often lack the flexibility to explore modelling uncertainties, the usability and modularity to explore a wide rage of modelling assumptions, the performance necessary to explore a large parameter space and process modern day large datasets, or documentation, and thus consistency, and reproducibility across a range datasets.

Synthesizer addresses these shortcomings by offering:

- Flexibility: Anything that could potentially be changed by the user is designed from the ground up to either be variable or swappable with an alternative. This means that users can easily vary everything in a reproducible way without needing to modify the core code.

- Performance: Computationally intensive operations are optimized by employing C extensions with OpenMP threading. Without this performance the aforementioned flexbility is moot, only by coupling flexibility with the performance to utilise it can we explore large parameter spaces in a reasonable time.

- Modularity: Synthesizer is object-orientated, with a focus on decoupled classes that can be specialised and then swapped out at will. This modularity, in conjunction with a reliance on templating and dependency injection (see Emission Models below), is what enables Synthesizer's flexibility.

- Extensibility: Extensive documentation and a clear API enables users to extend the package with their own calculations, and parametrisations. From the beginning, Synthesizer has been designed to be expanded to fit the needs of all users, even as astronomy and astrophysics evolves.

Synthesizer's design facilitates apples-to-apples comparisons between simulations and observations, exhaustive tests of modelling parameter impact, promotes reproducible science, and enables the forward modelling of large datasets previously considered impractical.

# Implementation overview

Synthesizer is structured around a set of core abstractions, here we give a brief outline of these abstractions and a link to the documentation for each.

- **Components**: Represent [stars](https://synthesizer-project.github.io/synthesizer/components/stars.html), [gas](https://synthesizer-project.github.io/synthesizer/components/gas.html), and [black holes](https://synthesizer-project.github.io/synthesizer/components/blackholes.html), encapsulating physical properties, and emission and observable generator methods. For more details see the [components documentation](https://synthesizer-project.github.io/synthesizer/components/components.html).
- **Galaxies**: Combine multiple components into a single object for cohesive calculations with all components. For more detials see the [galaxies documentation](https://synthesizer-project.github.io/synthesizer/galaxy/galaxy.html).
- **Emission Grids**: N-dimensional lookup tables of precomputed spectra and lines. Precomputed grids are available for models including BC03 [@bc03], BPASS [@bpass], FSPS (@fsps1, @fsps2), Maraston (@maraston05, @newman25), all reprocessed using Cloudy [@cloudy]. Users can also generate custom grids via the accompanying [grid-generation package](https://github.com/synthesizer-project/grid-generation). For more details see the [grids documentation](https://synthesizer-project.github.io/synthesizer/grids/grids.html).
- **Emission Models**: Modular templates defining the process of producing emissions from components. These models can be used to extract, generate, transform, or combine emissions. These are the backbone of Synthesizer's flexibility and modularity. For more details see the [emission models documentation](https://synthesizer-project.github.io/synthesizer/emission_models/emission_models.html).
- **Emissions**: The output of combining components with an emission model. These emissions are either spectra stored in [`Sed` objects](https://synthesizer-project.github.io/synthesizer/sed/sed.html), or line emissions stored in [`LineCollection` objects](https://synthesizer-project.github.io/synthesizer/lines/lines.html).
- **Instruments**: Definitions of filters, resolutions, PSFs, and noise models to convert emissions into photometry, spectroscopy, images, and data cubes. For more details see the [instruments documentation](https://synthesizer-project.github.io/synthesizer/instrumentation/instrument_example.html) and [filters documentation](https://synthesizer-project.github.io/synthesizer/filters/filters.html).
- **Observables**: Containers for the output spectra with observational effects ([`Sed` objects](https://synthesizer-project.github.io/synthesizer/sed/sed.html)), photometry ([`PhotometryCollection` objects](https://synthesizer-project.github.io/synthesizer/photometry/photometry.html)), images ([`Image` and `ImageCollection` objects](https://synthesizer-project.github.io/synthesizer/imaging/imaging.html)), and spectral data cubes ([`SpectralDataCube` objects](https://synthesizer-project.github.io/synthesizer/imaging/imaging.html)).

Synthesizer is hosted on [GitHub](https://github.com/synthesizer-project/synthesizer) and is available on [PyPI](https://pypi.org/project/cosmos-synthesizer/). The documentation is available through [ReadTheDocs](https://synthesizer-project.github.io/synthesizer/).

# Acknowledgements

We acknowledge the use of the following software packages in this work: [Astropy](https://www.astropy.org/) [@astropy], [Matplotlib](https://matplotlib.org/) [@matplotlib], [NumPy](https://numpy.org/) [@numpy], [SciPy](https://www.scipy.org/) [@scipy], and [OpenMP](https://www.openmp.org/).

WJR, APV, and SMW acknowledge support from the Sussex Astronomy Centre STFC Consolidated Grant (ST/X001040/1). CCL acknowledges support from a Dennis Sciama fellowship funded by the University of Portsmouth for the Institute of Cosmology and Gravitation. ...

# References
