Grids
*****

Introduction
============

Most of the functionality of synthesizer is reliant on *grid files*. These are typically precomputed multi-dimensional arrays of spectra (and lines) from Stellar Population Synthesis (SPS) models for a range of ages and metallicities, and potentially other parameters (see below).
Grids can also represent the emission from other sources, e.g. active galactic nuclei.

There is a low-resolution test grid available via the ``synthesizer-download`` command line tool, but for actual projects you will need to download one or more full production grids `from Box <https://sussex.box.com/v/SynthesizerProductionGrids>`_. See details below on where on your system to download these grids and how to load them. 

The Grid Directory
------------------

All synthesizer grids should be stored in a separate directory somewhere on your system. For example, we can create a folder:

.. code-block:: bash

    mkdir /our/synthesizer/data_directory/synthesizer_data/

Within this we will additionally create another directory to hold our grids:

.. code-block:: bash

    mkdir /our/synthesizer/data_directory/synthesizer_data/grids

If you wish, you can set this grid directory as an environment variable.

Pre-Computed Grids
==================

A goal of synthesizer is to be **flexible**.
With this in mind, we have generated a variety of grids for different SPS models, initial mass functions (IMFs), and photoionisation modelling assumptions.

.. _grid-naming:

Grid naming
-----------

The naming of grids currently follows this specification::

    {sps_model}-{sps_version}-{sps_variant}_{imf_type}-{mass_boundaries}-{slopes}_{photoionisation_code}-{photoionisation_code_version}-{photoionisation_parameters} 

e.g. ::

    bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c17.03 

specifies that the grid is constructed using v2.2.1 of the `Binary Population and Spectral Synthesis <https://bpass.auckland.ac.nz/>`_ (BPASS) SPS model for the binary (bin) variant. This grid assumes the Chabrier (2003) IMF between 0.1 and 300 Msol. Photoionisation modelling is performed using v17.03 of the `cloudy <https://gitlab.nublado.org/cloudy/cloudy>`_ photoionisation code assuming our default assumptions.


Initial Mass Function
---------------------

Grids are constructed using various initial mass functions (IMFs), often depending on the availability in the specific SPS model.
In most cases we recommend using the Chabrier (2003) IMF since this is available for most SPS models, allowing a like-for-like comparison.
If you're interested in exploring the systematic impact of changing the IMF, broken power law (bpl) IMFs may be suitable. These are named e.g. ::

    {imf_type}-{mass_boundaries}-{slopes}

e.g. for a Salpeter (1955) IMF (slope=2.35) between 0.1 and 100 Msol we would have ::

    bpl-0.1,100-2.35

A more complex IMF, for example with two power-laws (2.0, 2.35) separated at 1 Msol, would have ::

    bpl-0.1,1.0,100-2.0,2.35

If an IMF you need is missing, please let us know by raising a feature request through an `issue <https://github.com/synthesizer-project/synthesizer/issues/new?assignees=&labels=&projects=&template=feature_request.md&title=>`_.


Photoionisation modelling
-------------------------
All the photoionisation modelling in synthesizer currently uses the `cloudy <https://gitlab.nublado.org/cloudy/cloudy>`_ photoionisation code. Our default assumptions are:

* `log10(U)=-2`


Common variants
---------------

* `resolution:0.1` outputs the spectra at 10x higher resolution than the `cloudy` default. Useful for looking at various absorption line indices. 
* `log10U:X` assumes a different ionisation parameter.


Higher-dimensionality grids
---------------------------
Most SPS grids are two-dimensional, with the dimensions being `log10(age)` and `metallicity`. However synthesizer can utilise grids with higher dimensionality e.g. including varying alpha-abundance, or photoionisation parameters (e.g. `U`).

By default, certain models (e.g., parametric stars) aren't set up to handle higher dimensionality, though this may change in a future version. 
For now, we provide the functionality to handle these grids by "collapsing" over the additional axes. 
More details on this are provided in the `grids_example <grids_example>` notebook.

Grid list
=========

Below are examples of the pre-computed grids available in the `Box <https://sussex.app.box.com/v/SynthesizerProductionGrids>`_.

.. collapse:: Bruzual & Charlot (2003, BC03)

    * Chabrier (2003) IMF
        - bc03-2003-padova00_chabrier03-0.1,100

.. collapse:: 2016 update of Bruzual & Charlot (2003)

    * The BaSel variant
        - Chabrier (2003) IMF
            + bc03-2016-BaSeL_chabrier-0.1,100
            + bc03-2016-BaSeL_chabrier-0.1,100_cloudy-c23.01-sps
    
    * The Miles variant
        - Chabrier (2003) IMF
            + bc03-2016-Miles_chabrier03-0.1,100 
            + bc03-2016-Miles_chabrier-0.1,100_cloudy-c23.01-sps
    
    * The Stelib variant
        - Chabrier (2003) IMF
            + bc03-2016-Stelib_chabrier-0.1,100
            + bc03-2016-Stelib_chabrier-0.1,100_cloudy-c23.01-sps

.. collapse:: Binary Population and Spectral Synthesis (BPASS) v2.2.1

    `Binary Population and Spectral Synthesis <https://bpass.auckland.ac.nz/>`_ 

    * Binary variant
        - Broken power-law IMF 
            + bpass-2.2.1-bin_bpl-0.1,1.0,100.0-1.3,2.35
            + bpass-2.2.1-bin_bpl-0.1,1.0,100.0-1.3,2.0_cloudy-c23.01-sps
            + bpass-2.2.1-bin_bpl-0.1,1.0,300.0-1.3,2.35_cloudy-c23.01-sps
            
        - Chabrier (2003) IMF
            + bpass-2.2.1-bin_chabrier03-0.1,100.0_cloudy-c23.01-sps
            + bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps
            + bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps-fixed_ionisation_parameter
    
    * Single star variant variant
        - Broken power-law IMF
            + bpass-2.2.1-sin_bpl-0.1,1.0,300.0-1.3,2.7
            + bpass-2.2.1-sin_bpl-0.1,1.0,300.0-1.3,2.0_cloudy-c23.01-sps
            + bpass-2.2.1-sin_bpl-0.1,1.0,300.0-1.3,2.35_cloudy-c23.01-sps
            
        - Chabrier (2003) IMF
            + bpass-2.2.1-sin_chabrier03-0.1,100.0
            + bpass-2.2.1-sin_chabrier03-0.1,100.0_cloudy-c23.01-sps
            + bpass-2.2.1-sin_chabrier03-0.1,300.0_cloudy-c23.01-sps



.. collapse:: Binary Population and Spectral Synthesis (BPASS) v2.3

    `Binary Population and Spectral Synthesis <https://bpass.auckland.ac.nz/>`_ 
    
    * Binary variant
        - Broken power-law IMF
            + bpass-2.3-bin_bpl-0.1,1.0,300.0-1.3,2.35
            + bpass-2.3-bin_bpl-0.1,1.0,300.0-1.3,2.35_alpha0.0_cloudy-c23.01-sps
            + bpass-2.3-bin_bpl-0.1,1.0,300.0-1.3,2.35_alpha0.4_cloudy-c23.01-sps
            + bpass-2.3-bin_bpl-0.1,1.0,300.0-1.3,2.35_alpha0.6_cloudy-c23.01-sps



.. collapse:: Flexible Stellar Population Synthesis (FSPS) v3.2
    
    * Broken power-law IMF 
        - fsps-3.2-mistmiles_bpl-0.08,0.5,1,120-1.3,2.3,2.1_cloudy-c23.01-sps
        - fsps-3.2-mistmiles_bpl-0.08,0.5,1,120-1.3,2.3,3.0_cloudy-c23.01-sps
        - fsps-3.2-mistmiles_bpl-0.08,0.5,1,120-1.3,2.3,2.7_cloudy-c23.01-sps
    
    * Chabrier (2003) IMF
        - fsps-3.2-mistmiles_chabrier03-50,120
        - fsps-3.2-mistmiles_chabrier03-0.08,100_cloudy-c23.01-sps
        - fsps-3.2-mistmiles_chabrier03-0.08,5_cloudy-c23.01-sps

.. collapse:: Maraston models
    
    * Broken power-law IMF  
        - maraston05-rhb_bpl-0.1,100-2.35
        - maraston05-rhb_bpl-0.1,100-2.35_cloudy-c23.01-sps
    
    * Kroupa IMF 
        - maraston13_kroupa-0.1,100
        - maraston24-Tenc_0.00_salpeter-0.1,100
        - maraston24-Tenc40_kroupa-0.1,100_cloudy-c23.01-sps


Exploring Grids
===============

Once you've downloaded a grid you can get started here:

.. toctree::
   :maxdepth: 1

   grids_example


Creating your own grids
=======================

For advanced users, synthesizer contains scripts for creating your own grids from popular SPS codes, and running these through CLOUDY.
We provide scripts for doing this in the `grid-generation` repository.
Details are provided `here <../advanced/creating_grids>`_.
You will need a working installation of synthesizer for these scripts to work, as well as other dependencies for specific codes (e.g. `CLOUDY`, `python-FSPS`).
Please reach out to us if you have questions about the pre-computed grids or grid creation.
