.. _grids:
Grids
*****

Most of the functionality of `synthesizer` is reliant on Stellar Population Synthesis (SPS) grids. These are precomputed grids of spectra (and lines) for a range of ages and metallicities.

Most users will use pre-computed grids, available from dropbox `here <https://www.dropbox.com/sh/ipo6pox1sigjnqt/AADXfPvu7NbiWYiSGiooC_L0a?dl=0>`_. See details below on where to download these grids and how to load them. 

For advanced users, Synthesizer contains scripts for creating your own grids from popular SPS codes, and running these through CLOUDY. These are contained within the `generate_grids` directory of synthesizer. You will need a working installation of synthesizer for these scripts to work, as well as other dependencies for specific codes (e.g. CLOUDY, python-FSPS). Please reach out to us if you have questions about the pre-computed grids or grid creation.


**The Grid Directory**
All `synthesizer` grids should be stored in a separate directory somewhere on your system. For example, we can create a folder::

    mkdir /our/synthesizer/data_directory/synthesizer_data/

Within this we will additionally create another directory to hold our grids::

    mkdir /our/synthesizer/data_directory/synthesizer_data/grids

Pre-computed grids can be downloaded `here <https://www.dropbox.com/sh/ipo6pox1sigjnqt/AADXfPvu7NbiWYiSGiooC_L0a?dl=0>`_. We recommend downloading the following for the examples below:

* bc03_chabrier03_cloudy-v17.03_log10Uref-2.h5
* bc03_chabrier03.h5

If you wish, you can set this grid directory as an environment variable.


**Loading & Using Grids**

.. toctree::
   :maxdepth: 2

   explore_grid
   grid_lines

