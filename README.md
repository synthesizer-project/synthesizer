# Synthesizer

[![workflow](https://github.com/flaresimulations/synthesizer/actions/workflows/python-app.yml/badge.svg)](https://github.com/flaresimulations/synthesizer/actions)
[![Documentation Status](https://github.com/flaresimulations/synthesizer/actions/workflows/publish_docs.yml/badge.svg)](https://github.com/flaresimulations/synthesizer/actions)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/flaresimulations/synthesizer/blob/main/docs/CONTRIBUTING.md)

Synthesizer is a python package for generating synthetic astrophysical spectra. It is intended to be modular, flexible and fast.

Here are just some examples of what synthesizer can do:
- generate spectra for parametric star formation and metal enrichment histories
- generate spectra for galaxies from particle-based cosmological hydrodynamic simulations
- measure spectral diagnostics for given spectra
- easily compare stellar population synthesis models
- apply screen dust models to intrinsic spectra

Read the documentation [here](https://synthesizer-docs.readthedocs.io/en/latest/).

## Contributing

Please see [here](docs/CONTRIBUTING.md) for contribution guidelines.

## Citation & Acknowledgement

A code paper is currently in preparation. For now please cite [Vijayan et al. 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.501.3289V/abstract) if you use the functionality for producing photometry, and [Wilkins et al. 2020](https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.6079W/abstract) if you use the line emission functionality.

    @article{10.1093/mnras/staa3715,
      author = {Vijayan, Aswin P and Lovell, Christopher C and Wilkins, Stephen M and Thomas, Peter A and Barnes, David J and Irodotou, Dimitrios and Kuusisto, Jussi and Roper, William J},
      title = "{First Light And Reionization Epoch Simulations (FLARES) -- II: The photometric properties of high-redshift galaxies}",
      journal = {Monthly Notices of the Royal Astronomical Society},
      volume = {501},
      number = {3},
      pages = {3289-3308},
      year = {2020},
      month = {11},
      issn = {0035-8711},
      doi = {10.1093/mnras/staa3715},
      url = {https://doi.org/10.1093/mnras/staa3715},
      eprint = {https://academic.oup.com/mnras/article-pdf/501/3/3289/35651856/staa3715.pdf},
    }

    @article{10.1093/mnras/staa649,
      author = {Wilkins, Stephen M and Lovell, Christopher C and Fairhurst, Ciaran and Feng, Yu and Matteo, Tiziana Di and Croft, Rupert and Kuusisto, Jussi and Vijayan, Aswin P and Thomas, Peter},
      title = "{Nebular-line emission during the Epoch of Reionization}",
      journal = {Monthly Notices of the Royal Astronomical Society},
      volume = {493},
      number = {4},
      pages = {6079-6094},
      year = {2020},
      month = {03},
      issn = {0035-8711},
      doi = {10.1093/mnras/staa649},
      url = {https://doi.org/10.1093/mnras/staa649},
      eprint = {https://academic.oup.com/mnras/article-pdf/493/4/6079/32980291/staa649.pdf},
    }

## Licence

[GNU General Public License v3.0](https://github.com/flaresimulations/synthesizer/blob/main/LICENSE.md)

## Authors

See [here](AUTHORS.rst) for details on the core authors.
