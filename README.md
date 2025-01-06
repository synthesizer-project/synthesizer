# Synthesizer

<img src="docs/source/img/synthesizer_logo.png" align="right" width="140px"/>

[![workflow](https://github.com/flaresimulations/synthesizer/actions/workflows/python-app.yml/badge.svg)](https://github.com/flaresimulations/synthesizer/actions)
[![Documentation Status](https://github.com/flaresimulations/synthesizer/actions/workflows/publish_docs.yml/badge.svg)](https://flaresimulations.github.io/synthesizer/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/flaresimulations/synthesizer/blob/main/docs/CONTRIBUTING.md)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)



Synthesizer is a python package for generating synthetic astrophysical observables. It is modular, flexible and fast.

Read the documentation [here](https://flaresimulations.github.io/synthesizer/).

## Getting Started

First clone the latest version of `synthesizer`

    git clone https://github.com/flaresimulations/synthesizer.git

To install, enter the `synthesizer` directory and install with pip.

    cd synthesizer
    pip install .
    
We also provide optional dependency sets for development (``dev``), testing (``test``), and building the documentation (``docs``) should you ever needed them. To install all dependancies simply run the following (or delete as appropriate to get a specific subset):

    pip install .[dev,test,docs]

Make sure you stay up to date with the latest versions through git:

    git pull origin main

### Getting test data

If you want to run some of the examples or like some data for development purposes you'll need to download the test data. This can be done using the download tool on the command line. Run the following at the root of the Synthesizer repo,

```bash
synthesizer-download --test-grids --dust-grid -d tests/test_grid
```
This command will store the SPS, AGN, and dust grids in the `tests` directory where all examples expect them to be.

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
