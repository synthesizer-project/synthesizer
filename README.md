# Synthesizer

<img src="https://raw.githubusercontent.com/synthesizer-project/synthesizer/main/docs/source/img/synthesizer_logo.png" align="right" width="140px"/>

[![workflow](https://github.com/synthesizer-project/synthesizer/actions/workflows/python-app.yml/badge.svg)](https://github.com/synthesizer-project/synthesizer/actions)
[![Documentation Status](https://github.com/synthesizer-project/synthesizer/actions/workflows/static.yml/badge.svg)](https://synthesizer-project.github.io/synthesizer/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/synthesizer-project/synthesizer/blob/main/docs/CONTRIBUTING.md)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![PyPI version](https://img.shields.io/pypi/v/cosmos-synthesizer.svg)](https://pypi.org/project/cosmos-synthesizer/)

<!--
This will also display the number of downloads but lets hide for now...
[![Downloads](https://img.shields.io/pypi/dm/cosmos-synthesizer.svg)](https://pypi.org/project/cosmos-synthesizer/)
-->

Synthesizer is a Python package for generating synthetic astrophysical observables. It is modular, flexible, extensible and fast.

Read the documentation [here](https://synthesizer-project.github.io/synthesizer/).

## Getting Started

The latest stable release of Synthesizer can be installed directly using pip,

```bash
pip install cosmos-synthesizer
```

Please refer to the [installation documentation](https://synthesizer-project.github.io/synthesizer/getting_started/installation.html) for further information.

**Note**: We do not currently support Windows, to use Synthesizer on Windows please install the Windows Subsystem for Linux (WSL).

Various configuration options can also be set at installation (see [here](https://synthesizer-project.github.io/synthesizer/advanced/config_options.html)).

## Getting Grids

In most use cases you will need a grid of theoretical spectra. Premade grids can be downloaded from the [grids data server](https://www.dropbox.com/scl/fo/3n8v3o4m85b0t8fl8pm0n/h?rlkey=9x4cijjnmvw5m6plnyovywuva&e=1&dl=0).

Note that you can also create your own grids using (or adapting) the [`grid-generation` repo](https://github.com/synthesizer-project/grid-generation).

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

[GNU General Public License v3.0](https://github.com/synthesizer-project/synthesizer/blob/main/LICENSE.md)
