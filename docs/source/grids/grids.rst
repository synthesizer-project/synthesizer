.. _grids:
Grids
*****

Introduction
============

Most of the functionality of `synthesizer` is reliant on Stellar Population Synthesis (SPS) grids. These are precomputed grids of spectra (and lines) for a range of ages and metallicities, and potentially other parameters (see below)

There is a limited, low-resolution, test grid built into to `synthesizer` but for actual projects you will need to download one or more full production grid from dropbox `here <https://www.dropbox.com/scl/fo/x0ymq0qcor7mdoigpqpsp/h?rlkey=pax3p4f6spln30j2owkyq7vj7&dl=0>`_. See details below on where to download these grids and how to load them. 

For advanced users, Synthesizer contains scripts for creating your own grids from popular SPS codes, and running these through CLOUDY. These are contained within the `generate_grids` directory of synthesizer. You will need a working installation of synthesizer for these scripts to work, as well as other dependencies for specific codes (e.g. CLOUDY, python-FSPS). Please reach out to us if you have questions about the pre-computed grids or grid creation.


The Grid Directory
------------------

All `synthesizer` grids should be stored in a separate directory somewhere on your system. For example, we can create a folder::

    mkdir /our/synthesizer/data_directory/synthesizer_data/

Within this we will additionally create another directory to hold our grids::

    mkdir /our/synthesizer/data_directory/synthesizer_data/grids

If you wish, you can set this grid directory as an environment variable.

Available Grids
===============

A goal of `synthesizer` is to be **flexible**. With this in mind we have generated a variety of grids for different stellar population synthesis (SPS) models, initial mass functions (IMFs), and photoionisation modelling assumptions.

Grid naming
-----------

The naming of grids currently follows this approach::

    {sps_model}-{sps_version}-{sps_variant}_{imf_type}-{mass_boundaries}-{slopes}_{photoionisation_code}-{photoionisation_code_version}-{photoionisation_parameters} 

e.g. ::

    bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c17.03 

Is the grid constructed using v2.2.1 of the `Binary Population and Spectral Synthesis <https://bpass.auckland.ac.nz/>`_ (BPASS) SPS model for the binary (bin) variant. This grid assumes the Chabrier (2003) IMF between 0.1 and 300 Msol. Photoionisation modelling is performed using v17.03 of the `cloudy <https://trac.nublado.org>`_ photoionisation code assuming our default assumptions.


Initial Mass Function
---------------------

Grids are constructed using various initial mass functions (IMFs), often depending on the availability in the specific SPS model. In most cases we recommend using the Chabrier (2003) IMF since this is available for most SPS models. If you're interest in exploring the systematic impact of changing the IMF  broken power law (bpl) IMFs may be suitable. These are named e.g. ::

    {imf_type}-{mass_boundaries}-{slopes}

e.g. for a Salpeter (1955) IMF (slope=2.35) between 0.1 and 100 Msol we would have ::

    bpl-0.1,100-2.35

For a more complext IMF, for example with two power-laws (2.0, 2.35) separated at 1 Msol we would have ::

    bpl-0.1,1.0,100-2.0,2.35

If an IMF you need is missing please let us know.







Photoionisation modelling
-------------------------
All the photoionisation modelling in `synthesizer` currently uses the `cloudy <https://trac.nublado.org>`_ photoionsation code. Our default assumptions are:

* `log10(U)=-2`

Our photoionisation modelling is described in more detail in :ref:`grids-photoionisation`.


Common variants
^^^^^^^^^^^^^^^

* `resolution:0.1` outputs the spectra at 10x higher resolution than the `cloudy` default. Useful for looking at various absorption line indices. 
* `log10U:X` assumes a different ionisation parameter.


Higher-dimensionality grids
---------------------------
Most SPS grids are two-dimensional with the dimensions being `log10(age)` and `metallicty`. However `synthesizer` can utlilise grids with higher dimensionality e.g. including varying alpha-abundance, or photoionisation parameters (e.g. `U`).



Grid list
=========

**Bruzual & Charlot (2003, BC03)**

* Chabrier (2003) IMF
    - bc03_chabrier03-0.1,100 
    - bc03_chabrier03-0.1,100_cloudy-c17.03 
    - bc03_chabrier03-0.1,100_cloudy-c17.03_resolution:0.1 


**2016 update of Bruzual & Charlot (2003)**

* The BaSel variant
    - Chabrier (2003) IMF
        + bc03-2016-BaSeL_chabrier03-0.1,100 
        + bc03-2016-BaSeL_chabrier03-0.1,100_cloudy-c17.03 
        + bc03-2016-BaSeL_chabrier03-0.1,100_cloudy-c17.03_resolution:0.1 

* The Miles variant
    - Chabrier (2003) IMF
        + bc03-2016-Miles_chabrier03-0.1,100 
        + bc03-2016-Miles_chabrier03-0.1,100_cloudy-c17.03 
        + bc03-2016-Miles_chabrier03-0.1,100_cloudy-c17.03_resolution:0.1 

* The Stelib variant
    - Chabrier (2003) IMF
        + bc03-2016-Stelib_chabrier03-0.1,100 
        + bc03-2016-Stelib_chabrier03-0.1,100_cloudy-c17.03 
        + bc03-2016-Stelib_chabrier03-0.1,100_cloudy-c17.03_resolution:0.1 

**Binary Population and Spectral Synthesis (BPASS) v2.2.1** 
`Binary Population and Spectral Synthesis <https://bpass.auckland.ac.nz/>`_ 

* Binary variant
    - Broken power-law IMF
        + bpass-2.2.1-bin_bpl-0.1,1.0,100.0-1.3,2.0 
        + bpass-2.2.1-bin_bpl-0.1,1.0,100.0-1.3,2.0_cloudy-c17.03 
        + bpass-2.2.1-bin_bpl-0.1,1.0,100.0-1.3,2.35 
        + bpass-2.2.1-bin_bpl-0.1,1.0,100.0-1.3,2.35_cloudy-c17.03 
        + bpass-2.2.1-bin_bpl-0.1,1.0,100.0-1.3,2.7 
        + bpass-2.2.1-bin_bpl-0.1,1.0,100.0-1.3,2.7_cloudy-c17.03 
        + bpass-2.2.1-bin_bpl-0.1,1.0,300.0-1.3,2.0 
        + bpass-2.2.1-bin_bpl-0.1,1.0,300.0-1.3,2.0_cloudy-c17.03 
        + bpass-2.2.1-bin_bpl-0.1,1.0,300.0-1.3,2.0_cloudy-c17.03_resolution:0.1 
        + bpass-2.2.1-bin_bpl-0.1,1.0,300.0-1.3,2.35 
        + bpass-2.2.1-bin_bpl-0.1,1.0,300.0-1.3,2.35_cloudy-c17.03 
        + bpass-2.2.1-bin_bpl-0.1,1.0,300.0-1.3,2.35_cloudy-c17.03_resolution:0.1 
        + bpass-2.2.1-bin_bpl-0.1,1.0,300.0-1.3,2.7 
        + bpass-2.2.1-bin_bpl-0.1,1.0,300.0-1.3,2.7_cloudy-c17.03 
        + bpass-2.2.1-bin_bpl-0.1,1.0,300.0-1.3,2.7_cloudy-c17.03_resolution:0.1 
    - Chabrier (2003) IMF
        + bpass-2.2.1-bin_chabrier03-0.1,100.0 
        + bpass-2.2.1-bin_chabrier03-0.1,100.0_cloudy-c17.03 
        + bpass-2.2.1-bin_chabrier03-0.1,100.0_cloudy-c17.03_resolution:0.1 
        + bpass-2.2.1-bin_chabrier03-0.1,300.0 
        + bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c17.03 
        + bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c17.03_resolution:0.1 

* Single star variant variant
    - Broken power-law IMF
        + bpass-2.2.1-sin_bpl-0.1,1.0,100.0-1.3,2.0 
        + bpass-2.2.1-sin_bpl-0.1,1.0,100.0-1.3,2.35 
        + pass-2.2.1-sin_bpl-0.1,1.0,100.0-1.3,2.7 
        + bpass-2.2.1-sin_bpl-0.1,1.0,300.0-1.3,2.0 
        + bpass-2.2.1-sin_bpl-0.1,1.0,300.0-1.3,2.35 
        + bpass-2.2.1-sin_bpl-0.1,1.0,300.0-1.3,2.7 
    - Chabrier (2003) IMF
        + bpass-2.2.1-sin_chabrier03-0.1,100.0 
        + bpass-2.2.1-sin_chabrier03-0.1,300.0 

**Binary Population and Spectral Synthesis (BPASS) v2.3** 
`Binary Population and Spectral Synthesis <https://bpass.auckland.ac.nz/>`_ 

* Binary variant
    - Chabrier (2003) IMF
        + bpass-2.3-bin_chabrier03-0.1,300.0 
        + bpass-2.3-bin_chabrier03-0.1,300.0_alpha-0.2 
        + bpass-2.3-bin_chabrier03-0.1,300.0_alpha-0.2_cloudy-c17.03_resolution:0.1 
        + bpass-2.3-bin_chabrier03-0.1,300.0_alpha0.0 
        + bpass-2.3-bin_chabrier03-0.1,300.0_alpha0.2 
        + bpass-2.3-bin_chabrier03-0.1,300.0_alpha0.2_cloudy-c17.03_resolution:0.1 
        + bpass-2.3-bin_chabrier03-0.1,300.0_alpha0.4 
        + bpass-2.3-bin_chabrier03-0.1,300.0_alpha0.4_cloudy-c17.03_resolution:0.1 
        + bpass-2.3-bin_chabrier03-0.1,300.0_alpha0.6 
        + bpass-2.3-bin_chabrier03-0.1,300.0_alpha0.6_cloudy-c17.03_resolution:0.1 


**Flexible Stellar Population Synthesis (FSPS) v3.2** 

* Broken power-law IMF - Note: several of these are currently missing.
    - fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,2.3 
    - fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,2.3_cloudy-c17.03 
    - fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,2.8 
    - fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,2.8_cloudy-c17.03 
    - fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,2.9 
    - fsps-3.2_bpl-0.08,0.5,1,120-1.3,2.3,3.0 

* Chabrier (2003) IMF
    - fsps-3.2_chabrier03-0.08,1 
    - fsps-3.2_chabrier03-0.08,10 
    - fsps-3.2_chabrier03-0.08,100 
    - fsps-3.2_chabrier03-0.08,100_cloudy-c17.03 
    - fsps-3.2_chabrier03-0.08,120 
    - fsps-3.2_chabrier03-0.08,120_cloudy-c17.03 
    - fsps-3.2_chabrier03-0.08,120_cloudy-c17.03_resolution:0.1 
    - fsps-3.2_chabrier03-0.08,2 
    - fsps-3.2_chabrier03-0.08,20 
    - fsps-3.2_chabrier03-0.08,5 
    - fsps-3.2_chabrier03-0.08,50 
    - fsps-3.2_chabrier03-0.08,50_cloudy-c17.03 
    - fsps-3.2_chabrier03-0.5,120 
    - fsps-3.2_chabrier03-0.5,120_cloudy-c17.03 
    - fsps-3.2_chabrier03-1,120 
    - fsps-3.2_chabrier03-1,120_cloudy-c17.03 
    - fsps-3.2_chabrier03-10,120 
    - fsps-3.2_chabrier03-10,120_cloudy-c17.03 
    - fsps-3.2_chabrier03-2,120 
    - fsps-3.2_chabrier03-2,120_cloudy-c17.03 
    - fsps-3.2_chabrier03-20,120 
    - fsps-3.2_chabrier03-20,120_cloudy-c17.03 
    - fsps-3.2_chabrier03-5,120 
    - fsps-3.2_chabrier03-50,120 
    - fsps-3.2_chabrier03-50,120_cloudy-c17.03 

**Maraston models** 

* Broken power-law IMF  
    - maraston-rhb_bpl-0.1,100-2.35 



Loading and using Grids
=======================

.. toctree::
   :maxdepth: 2
   
   photoionisation
   explore_grid
   plot_spectra
   plot_spectra_age
   plot_log10Q
   grid_lines
   
