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
   performance/performance
   advanced/advanced
   notebook_examples/cookbook
   auto_examples/index
   publications/publications
   API

Citation & Acknowledgement
--------------------------

A code paper is currently in preparation. For now please cite `Vijayan et al. 2021 <https://ui.adsabs.harvard.edu/abs/2021MNRAS.501.3289V/abstract>`_ if you use the functionality for producing photometry, and `Wilkins et al. 2020 <https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.6079W/abstract>`_ if you use the line emission functionality.

.. code-block:: bibtex

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

Contributing
------------

Please see `here <https://github.com/synthesizer-project/synthesizer/blob/main/docs/CONTRIBUTING.md>`_ for contribution guidelines.

Primary Contributors
---------------------

.. include:: ../../AUTHORS.rst

License
-------

Synthesizer is free software made available under the GNU General Public License v3.0. For details see the `LICENSE <https://github.com/synthesizer-project/synthesizer/blob/main/LICENSE.md>`_.

