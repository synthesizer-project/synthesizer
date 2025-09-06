# Synthesizer

<img src="https://raw.githubusercontent.com/synthesizer-project/synthesizer/main/docs/source/img/synthesizer_logo.png" align="right" width="140px"/>

[![workflow](https://github.com/synthesizer-project/synthesizer/actions/workflows/python-app.yml/badge.svg)](https://github.com/synthesizer-project/synthesizer/actions)
[![Documentation Status](https://github.com/synthesizer-project/synthesizer/actions/workflows/static.yml/badge.svg)](https://synthesizer-project.github.io/synthesizer/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/synthesizer-project/synthesizer/blob/main/docs/CONTRIBUTING.md)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![PyPI version](https://img.shields.io/pypi/v/cosmos-synthesizer.svg)](https://pypi.org/project/cosmos-synthesizer/)
[![status](https://joss.theoj.org/papers/cc4f37b2d2fec7d1bd48af22c01d78a7/status.svg)](https://joss.theoj.org/papers/cc4f37b2d2fec7d1bd48af22c01d78a7)
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

In most use cases you will need a grid of theoretical spectra. Premade grids can be downloaded from the [grids data server](https://sussex.box.com/v/SynthesizerProductionGrids).

Note that you can also create your own grids using (or adapting) the [`grid-generation` repo](https://github.com/synthesizer-project/grid-generation).

## Contributing

Please see [here](docs/CONTRIBUTING.md) for contribution guidelines.

## Citation & Acknowledgement

Please cite **both** of the following papers ([Lovell et al. 2025](https://ui.adsabs.harvard.edu/abs/2025arXiv250803888L/abstract), [Roper et al. 2025](https://ui.adsabs.harvard.edu/abs/2025arXiv250615811R/abstract)) if you use Synthesizer in your research:

    @ARTICLE{2025arXiv250803888L,
           author = {{Lovell}, Christopher C. and {Roper}, William J. and {Vijayan}, Aswin P. and {Wilkins}, Stephen M. and {Newman}, Sophie and {Seeyave}, Louise},
            title = "{Synthesizer: a Software Package for Synthetic Astronomical Observables}",
          journal = {arXiv e-prints},
         keywords = {Instrumentation and Methods for Astrophysics, Cosmology and Nongalactic Astrophysics, Astrophysics of Galaxies},
             year = 2025,
            month = aug,
              eid = {arXiv:2508.03888},
            pages = {arXiv:2508.03888},
    archivePrefix = {arXiv},
           eprint = {2508.03888},
     primaryClass = {astro-ph.IM},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv250803888L},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

    @ARTICLE{2025arXiv250615811R,
       author = {{Roper}, Will J. and {Lovell}, Christopher and {Vijayan}, Aswin and {Wilkins}, Stephen and {Akins}, Hollis and {Berger}, Sabrina and {Sant Fournier}, Connor and {Harvey}, Thomas and {Iyer}, Kartheik and {Leonardi}, Marco and {Newman}, Sophie and {Pautasso}, Borja and {Perry}, Ashley and {Seeyave}, Louise and {Sommovigo}, Laura},
        title = "{Synthesizer: Synthetic Observables For Modern Astronomy}",
      journal = {arXiv e-prints},
     keywords = {Instrumentation and Methods for Astrophysics, Astrophysics of Galaxies},
         year = 2025,
        month = jun,
          eid = {arXiv:2506.15811},
        pages = {arXiv:2506.15811},
    archivePrefix = {arXiv},
           eprint = {2506.15811},
     primaryClass = {astro-ph.IM},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv250615811R},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }


## Licence

[GNU General Public License v3.0](https://github.com/synthesizer-project/synthesizer/blob/main/LICENSE.md)
