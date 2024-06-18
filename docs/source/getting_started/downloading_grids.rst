Downloading Grids
=================

Synthesizer relies on pre-computed grids of spectra from stellar population synthesis models. Full details are provided in the `Grids <../grids/grids>`_ section of the documentation; here we describe how to download a simple test grid, which will enable you to run the examples and start to explore synthesizer.


Downloading the Test grid
^^^^^^^^^^^^^^^^^^^^^^^^^

Synthesizer comes packaged with a simple test grid. This should not be used for production runs (please see `here <../grids/grids>`_ for details) but is sufficient for running the examples and understanding the code.

To download the test grid, `cd` into the root synthesizer directory and run the `synthesizer-download` helper as follows:

    cd synthesizer
    synthesizer-download --test-grids -d tests/test_grid/ --dust-grid
