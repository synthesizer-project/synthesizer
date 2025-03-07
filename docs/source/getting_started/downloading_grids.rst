Downloading Grids
=================

Synthesizer relies on pre-computed grids of spectra from stellar population synthesis models. Full details are provided in the `Grids <../grids/grids>`_ section of the documentation; here we describe how to download a simple test grid, which will enable you to run the examples and start to explore synthesizer. 

Downloading the test grid
^^^^^^^^^^^^^^^^^^^^^^^^^

Synthesizer contains a simple command line helper tool for downloading the test grid. This test grid should not be used for production runs (please see `here <../grids/grids>`_ for details) but is sufficient for running the `examples <auto_examples/index>`_ and understanding the code. It is based on the BPASS stellar population synthesis model, and contains the stellar emission in the UV--optical on an age--metallicity grid.

To download the test grid run the ``synthesizer-download`` helper as follows:

.. code-block:: bash

    cd synthesizer
    synthesizer-download --test-grids -d tests/test_grid/ --dust-grid

Here we have assumed you have cloned the synthesizer repository and are in the top level directory. The examples will expect the grids to be stored at ``tests/test_grid/`` within the Synthesizer top level directory. 

    If you have instead installed synthesizer via pip, you can download the test grid to a directory of your choice, but you will need to point to this directory in any examples you run.  

You'll also want to download some test data if you want to run any particle based examples. We provided some preprocessed TNG data from CAMELS. To download it run 

.. code-block:: bash

    synthesizer-download --camels-data -d tests/data/

This will download the data to the ``tests/data/`` directory, the same caveat described above for the test grid applies here when installing via pip. 
