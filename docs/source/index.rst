Synthesizer
^^^^^^^^^^^

Synthesizer is an open-source python package for generating synthetic astrophysical observables. It is modular, flexible, fast and extensible.

This documentation provides a broad overview of the various components in synthesizer and how they interact.
The `getting started guide <getting_started/getting_started>`_ contains download and installation instructions, as well as an overview of the code.

For detailed examples of what synthesizer can do, take a look at the `examples <auto_examples/index>`_ page.
A full description of the code base is provided in the `API <API>`_.

Contents
^^^^^^^^

.. toctree::
   :maxdepth: 2
   
   getting_started/getting_started
   emission_grids/grids
   galaxy_components/galaxy_components
   emissions/emissions
   emission_models/emission_models
   observatories/observatories 
   observables/observables
   pipeline/pipeline_example
   parallelism
   performance/performance
   advanced/advanced
   notebook_examples/cookbook
   auto_examples/index
   publications/publications
   API

Citation & Acknowledgement
--------------------------

Please cite **both** of the following papers (`Lovell et al. 2025 <https://astro.theoj.org/article/145766-synthesizer-a-software-package-for-synthetic-astronomical-observables>`_, `Roper et al. 2026 <https://ui.adsabs.harvard.edu/abs/2025arXiv250615811R/abstract>`_) if you use Synthesizer in your research:

.. code-block:: bibtex

      @article{Lovell2025Synthesizer,
      	author = {Lovell, Christopher C. and Roper, William J. and Vijayan, Aswin P. and Wilkins, Stephen M. and Newman, Sophie and Seeyave, Louise},
      	journal = {The Open Journal of Astrophysics},
      	doi = {10.33232/001c.145766},
      	year = {2025},
      	month = {oct 9},
      	publisher = {Maynooth Academic Publishing},
      	title = {Synthesizer: a {Software} {Package} for {Synthetic} {Astronomical} {Observables}},
      	volume = {8},
      }

      @article{Roper2026Synthesizer,
         author = {Roper, Will J. and Lovell, Christopher C. and Vijayan, Aswin and Wilkins, Stephen and Akins, Hollis and Berger, Sabrina and Sant Fournier, Connor and Harvey, Thomas and Iyer, Kartheik and Leonardi, Marco and Newman, Sophie and Pautasso, Borja and Perry, Ashley and Seeyave, Louise and Sommovigo, Laura and Punyasheel, Paurush and d'Hautefort, Adrien Aufan Stoffels and Rawlings, Alex},
         journal = {Journal of Open Source Software},
         doi = {10.21105/joss.09436},
         year = {2026},
         publisher = {Open Journals},
         title = {Synthesizer: Synthetic Observables for Modern Astronomy},
         volume = {11},
         number = {119},
         pages = {9436},
      }

Contributing
------------

Please see `here <https://github.com/synthesizer-project/synthesizer/blob/main/docs/CONTRIBUTING.md>`_ for contribution guidelines.

Primary Contributors
---------------------

.. include:: ../../AUTHORS.rst

License
-------

Synthesizer is free software made available under the GNU General Public License v3.0. For details see the `LICENSE <https://github.com/synthesizer-project/synthesizer/blob/main/LICENSE.md>`_.

