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
  - name: Christopher C. Lovell
    orcid: 0000-0001-7964-5933
    equal-contrib: true
    affiliation: 2
  - name: Aswin Vijayan
    orcid: 0000-0002-1905-4194
    equal-contrib: true
    affiliation: 1
  - name: Stephen Wilkins
    orcid: 0000-0003-3903-6935
    equal-contrib: true
    affiliation: 1
  - name: Hollis Akins
    orcid: 0000-0003-3596-8794
    affiliation: 3
  - name: Sabrina Berger
    orcid: 0000-0002-4064-7883
    affiliation: 4
  - name: Connor Sant Fournier
    orcid: 0009-0004-0771-4476
    affiliation: 5
  - name: Thomas Harvey
    orcid: 0000-0002-4130-636X
    affiliation: 6
  - name: Kartheik Iyer
    orcid: 0000-0001-9298-3523
    affiliation: 7
  - name: Marco Leonardi
    orcid: 0009-0008-3592-7830
    affiliation: 8
  - name: Sophie Newman
    orcid: 0009-0001-3422-3048
    affiliation: 2
  - name: Borja Pautasso
    orcid: 0009-0003-1839-591X
    affiliation: 1
  - name: Ashley Perry
    orcid: 0009-0001-0739-6162
    affiliation: 1
  - name: Louise Seeyave
    orcid: 0000-0002-7020-3079
    affiliation: 1
  - name: Laura Sommovigo
    orcid: 0000-0002-2906-2200
    affiliation: 9

affiliations:
  - name: Astronomy Centre, University of Sussex, Falmer, Brighton BN1 9QH, UK
    index: 1
    ror: 00ayhx656
  - name: Kavli Institute for Cosmology, University of Cambridge, Madingley Road, Cambridge CB3 0HA, UK
    index: 2
  - name: Institute of Astronomy, University of Cambridge, Madingley Road, Cambridge CB3 0HA, UK
    index: 3
  - name: Institute of Cosmology and Gravitation, University of Portsmouth, Burnaby Road, Portsmouth, PO1 3FX, UK
    index: 4
  - name: Department of Astronomy, The University of Texas at Austin, Austin, TX 78712, USA
    index: 5
  - name: School of Physics, University of Melbourne, Parkville, VIC 3010, Australia
    index: 6
  - name: Institute of Space Sciences and Astronomy, University of Malta, Msida MSD 2080, Malta
    index: 7
  - name: Jodrell Bank Centre for Astrophysics, University of Manchester, Oxford Road, Manchester M13 9PL, UK
    index: 8
  - name: Columbia Astrophysics Laboratory, Columbia University, 550 West 120th Street, New York, NY 10027, USA
    index: 9
  - name: Leiden Observatory, Leiden University, PO Box 9513, NL-2300 RA Leiden, The Netherlands
    index: 10
  - name: Center for Computational Astrophysics, Flatiron Institute, 162 5th Ave, New York, NY 10010, USA
    index: 11

date: 17 June 2025
license: GPL-3.0
bibliography: synthesizer.bib
---

# Summary

Synthesizer is a fast, flexible, modular, and extensible Python package that empowers astronomers to turn theoretical galaxy models into realistic synthetic observations - including spectra, photometry, images, and spectral cubes - with a focus on interchangeable modelling assumptions. By offloading computationally intensive tasks to threaded C++ extensions, Synthesizer delivers both simplicity and speed, enabling rapid forward-modelling workflows without requiring users to manage low-level data processing and computational details.

# Statement of need

Producing synthetic observations from simulations has a long history in astronomy [e.g. @Torrey2015], but the availability of public, general-purpose tools for this task has only recently grown. Comparing theoretical models of galaxy formation with observations traditionally relies on two main approaches, both translating theoretical models into the observer space (a technique known as forward modelling). The first uses computationally expensive dust radiative transfer codes [e.g. @SKIRT; @SUNRISE; @POWDERDAY]; these codes are typically computationally expensive, prioritising fidelity. The second uses simpler, bespoke pipelines that sacrifice some physical fidelity to generate observables rapidly from large datasets [e.g. @Fortuni2023; @Marshall2025; @FLARESIV; @FLARESII; @SYNTHOBS]. Synthesizer is neither. It instead provides the tools to construct standardized, flexible pipelines that prioritize performance and modularity while enabling exploration of modelling assumptions.

Simplified inverse modelling approaches, such as SED fitting [e.g. @EAZY; @BAGPIPES; @PROSPECTOR] work in the opposite direction, translating observables into intrinsic physical quantities. However, these methods can introduce biases and uncertainties from both observational effects and model assumptions. Compounding these uncertainties is the fact that converged inverse modelling techniques are costly in their own right, necessitating a simplified parameter space to ensure convergence in a reasonable time. Forward modelling is therefore becoming increasingly important not only for probing the validity of theoretical models, but also for quantifying the uncertainties in the modelling assumptions themselves.

However, existing forward modelling tools often lack the flexibility to explore modelling uncertainties, the usability and modularity to explore a wide range of modelling assumptions, and the performance necessary to explore a large parameter space and process modern-day large datasets. Furthermore, they frequently lack comprehensive documentation, hindering consistency, and reproducibility across a range of datasets.

Synthesizer addresses these shortcomings by offering:

- Flexibility: Anything that could be changed by the user is explicitly designed to be variable (for a quantitative model parameter) or exchangeable (for a qualitative modelling choice). This means that users can easily vary everything in a reproducible way, without needing to modify the core code.

- Performance: Computationally intensive operations are optimised by employing C extensions with OpenMP threading. Without this performance, the aforementioned flexibility is moot; only by coupling flexibility with the performance to utilise it can we explore large, high-dimensional parameter spaces in a reasonable time.

- Modularity: Synthesizer is object-oriented, with a focus on decoupled classes that can be specialised and then swapped out at will. This modularity, in conjunction with a reliance on templating and dependency injection (see Emission Models below), is what enables Synthesizer's flexibility, as well as its application to a diverse range of astrophysical problems in both forward and inverse modelling

- Extensibility: Extensive documentation and a clear API enable users to extend the package with their own calculations, parameterisations and subclasses. From the beginning, Synthesizer has been designed to be expanded to fit the needs of all users, even as astronomy and astrophysics evolve.

Synthesizer's design facilitates apples-to-apples comparisons between simulations and observations [e.g. @FLARESXVIII], permits exhaustive tests of the impact of parameter choices [e.g. @LTU-ILI], enables the forward modelling of large datasets previously considered impractical [e.g. @LTU-LOVELL], and promotes open and reproducible science. Synthesizer's combination of modularity and performance is also critical for emerging inference techniques such as simulation-based inference (SBI), which require large training datasets of forward-modelled observables generated rapidly under flexible modelling assumptions, a regime where neither flexibility nor performance alone suffices. For example, @Harvey2025 use Synthesizer to inexpensively generate the training data needed for SBI-based SED fitting of galaxy photometry.

# Package overview

Synthesizer is structured around a set of core abstractions. Here we give a brief outline of these abstractions and a link to the documentation for each.

- **Components**: Represent [stars](https://synthesizer-project.github.io/synthesizer/galaxy_components/stars.html), [gas](https://synthesizer-project.github.io/synthesizer/galaxy_components/gas.html), and [black holes](https://synthesizer-project.github.io/synthesizer/galaxy_components/blackholes.html), encapsulating physical properties, and emission and emission generation methods. For more details, see the [components documentation](https://synthesizer-project.github.io/synthesizer/galaxy_components/galaxy_components.html#components).
- **Galaxies**: Combine multiple components into a single object, allowing for cohesive calculations with all components, taking account of their interdependencies. For more details, see the [galaxies documentation](https://synthesizer-project.github.io/synthesizer/galaxy_components/galaxy_components.html#the-galaxy-object).
- **Emission Grids**: N-dimensional lookup tables of precomputed spectra and lines. Precomputed grids are available for stellar population synthesis models, including BC03 [@bc03], BPASS [@bpass], FSPS (@fsps1, @fsps2), Maraston (@maraston05, @newman25), all reprocessed using Cloudy [@cloudy]. Grids of AGN emission can also be calculated and explored. Users can generate custom grids via the accompanying [grid-generation package](https://github.com/synthesizer-project/grid-generation). For more details, see the [grids documentation](https://synthesizer-project.github.io/synthesizer/emission_grids/grids.html).
- **Emission Models**: Modular templates defining the process of producing emissions from components. These models can be used to extract, generate, transform, or combine emissions. These are the backbone of Synthesizer's flexibility and modularity. For more details, see the [emission models documentation](https://synthesizer-project.github.io/synthesizer/emission_models/emission_models.html).
- **Emissions**: The output of combining components with an emission model. These emissions are either spectra stored in [`Sed` objects](https://synthesizer-project.github.io/synthesizer/emissions/emission_objects/sed_example.html), or line emissions stored in [`LineCollection` objects](https://synthesizer-project.github.io/synthesizer/emissions/emission_objects/lines_example.html).
- **Instruments**: Definitions of filters, resolutions, PSFs, and noise models to convert emissions into photometry, spectroscopy, images, and data cubes. For more details, see the [instruments documentation](https://synthesizer-project.github.io/synthesizer/observatories/observatories.html) and [filters documentation](https://synthesizer-project.github.io/synthesizer/observatories/filters.html).
- **Observables**: Containers for the output spectra with observational effects ([`Sed` objects](https://synthesizer-project.github.io/synthesizer/observables/spectroscopy/spectroscopy.html)), photometry ([`PhotometryCollection` objects](https://synthesizer-project.github.io/synthesizer/observables/photometry/photometry.html)), images ([`Image` and `ImageCollection` objects](https://synthesizer-project.github.io/synthesizer/observables/imaging/imaging.html)), and spectral data cubes ([`SpectralDataCube` objects](https://synthesizer-project.github.io/synthesizer/observables/spectral_data_cubes/spectral_data_cubes.html)).

Synthesizer is hosted on [GitHub](https://github.com/synthesizer-project/synthesizer) and is available on [PyPI](https://pypi.org/project/cosmos-synthesizer/). The documentation is available through [GitHub Pages](https://synthesizer-project.github.io/synthesizer/). A comprehensive description of Synthesizer's methodology and science validation is presented in the companion paper [@Lovell2025].

## Related packages

Several packages either overlap with Synthesizer’s functionality or complement it in end-to-end workflows. Some already interface with Synthesizer as plugins (e.g. SPS grids, PSF tools), while others are fully independent codes with conceptually similar goals:

- **Synthetic observation codes**: A number of codes produce synthetic observables from simulated galaxies, each targeting specific use cases. **FORECAST** [@Fortuni2023] and **GalaxyGenius** [@Zhou2025] generate mock images from hydrodynamical simulations for specific telescopes; **RealSim-IFS** [@Bottrell2022] and **SimSpin** [@Harborne2023] produce synthetic IFU datacubes; **py-ananke** [@Thob2024] creates stellar catalogs for Milky Way-like simulations; **pyMGal** [@Janulewicz2025] generates mock optical observations; **popkinmocks** [@Jethwa2023] produces mock IFU datacubes for stellar population and kinematic modelling; and **synphot** [@synphot] provides synthetic photometry utilities. While these share conceptual overlap with Synthesizer, they each target narrower use cases; Synthesizer aims to provide a general-purpose, modular framework spanning the full pipeline from theoretical models to multi-wavelength observables.
- **SPS & photoionisation**: Libraries for stellar spectra—**BC03** [@bc03], **FSPS** [@fsps1; @fsps2], **Maraston** [@maraston05], **BPASS** [@bpass]—paired with dust/nebular models, plus **Cloudy** [@cloudy] or **MAPPINGS** [@MAPPINGS] for reprocessing. These serve as inputs to Synthesizer’s emission grids.
- **Monte Carlo RT**: Photon–dust/gas simulators like **SKIRT** [@SKIRT], **Powderday** [@POWDERDAY], **Hyperion** [@hyperion]. These produce SEDs and images that can be ingested by Synthesizer.
- **PSF & instrument tools**: **STPSF** [@stpsf] (JWST, Roman, HST) and **GalSim** [@galsim] model telescope optics, detector effects, and noise. These integrate directly with Synthesizer’s instrument pipeline.
- **Pre/post-processing**: **YT** [@YT] for volumetric data analysis and visualization of simulation outputs; **Astroquery** [@astroquery] for automated querying of astronomical archives and catalogs; **Dense Basis** [@dense_basis] offers SED-/SFH-tailored basis functions.
- **Inverse modeling & SED fitting**: **EAZY** [@EAZY], **BAGPIPES** [@BAGPIPES], **PROSPECTOR** [@PROSPECTOR] extract galaxy properties from SEDs.

# Acknowledgements

We acknowledge the use of the following software packages in this work: [Astropy](https://www.astropy.org/) [@astropy], [unyt](https://unyt.readthedocs.io/en/stable/index.html) [@unyt], [Matplotlib](https://matplotlib.org/) [@matplotlib], [NumPy](https://numpy.org/) [@numpy], [SciPy](https://www.scipy.org/) [@scipy], and [OpenMP](https://www.openmp.org/) [@openmp].

WJR, APV, and SMW acknowledge support from the Sussex Astronomy Centre STFC Consolidated Grant (ST/X001040/1). CCL acknowledges support from a Dennis Sciama fellowship funded by the University of Portsmouth for the Institute of Cosmology and Gravitation. APV acknowledges support from the Carlsberg Foundation (grant no CF20-0534). SB is supported by the Melbourne Research Scholarship and N D Goldsworthy Scholarship. LS and SN are supported by an STFC studentship. This work was supported by the Simons Collaboration on “Learning the Universe”.

This work used the DiRAC@Durham facility managed by the Institute for Computational Cosmology on behalf of the STFC DiRAC HPC Facility (www.dirac.ac.uk). The equipment was funded by BEIS capital funding via STFC capital grants ST/K00042X/1, ST/P002293/1, ST/R002371/1 and ST/S002502/1, Durham University and STFC operations grant ST/R000832/1. DiRAC is part of the National e-Infrastructure.

# References
