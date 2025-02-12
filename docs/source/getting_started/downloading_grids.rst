Downloading Grids
=================

Synthesizer relies on pre-computed grids of spectra from stellar population synthesis models. Full details are provided in the `Grids <../grids/grids>`_ section of the documentation; here we describe how to download a simple test grid, which will enable you to run the examples and start to explore synthesizer.

Downloading the test grid
^^^^^^^^^^^^^^^^^^^^^^^^^

Synthesizer contains a simple command line helper tool for downloading the test grid. This test grid should not be used for production runs (please see `here <../grids/grids>`_ for details) but is sufficient for running the `examples <auto_examples/index>`_ and understanding the code. It is based on the BC03 stellar population synthesis model, and contains the stellar emission in the UV--optical on an age--metallicity grid.

To download the test grid, ``cd`` into the root synthesizer directory and run the ``synthesizer-download`` helper as follows:

.. code-block:: bash

    cd synthesizer
    synthesizer-download --test-grids -d tests/test_grid/ --dust-grid

You'll also want to download some test data if you want to run any particle based examples. We provided some preprocessed TNG data from CAMELS. To download it run 

.. code-block:: bash

    synthesizer-download --camels-data -d tests/data/
