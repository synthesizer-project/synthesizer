Creating Grids
**************

Introduction
============

Advanced users can create their own `synthesizer grids <grids/grids>`_. These can be intrinsic grids of stellar emission, generated from stellar population synthesis models, or grids post-processed through photoionisation codes such as `cloudy <https://trac.nublado.org>`_.

The code for creating custom grids is contained in a separate repository, `synthesizer-grids <https://github.com/flaresimulations/synthesizer-grids>`_.
You will need a working installation of synthesizer for these scripts to work, as well as other dependencies for specific codes (e.g. CLOUDY, python-FSPS). 

Grids should follow the naming convention where possible, see :ref:`grid-naming`.


Abundances
==========

The next page demonstrates how to modify the chemical abundance pattern of gas, stars and dust using the `abundances`` object, and use this when running `cloudy`.
