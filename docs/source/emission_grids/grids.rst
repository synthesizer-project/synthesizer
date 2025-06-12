Grids
*****

Introduction
============

Most of the functionality of Synthesizer is reliant on *grid files*. These are typically precomputed multi-dimensional arrays of spectra (and lines) from Stellar Population Synthesis (SPS) models for a range of ages and metallicities, and potentially other parameters (see below).
Grids can also represent the emission from other sources, e.g. active galactic nuclei.

There is a low-resolution test grid available via the ``synthesizer-download`` command line tool, but for actual projects you will need to download one or more full production grids from `Box <https://sussex.box.com/v/SynthesizerProductionGrids>`_. See details below on where on your system to download these grids and how to load them. 

Pre-Computed Grids
==================

Synthesizer was built on the ethos of being **flexible**.
With this in mind, we have generated a variety of grids for different SPS models, initial mass functions (IMFs), and photoionisation modelling assumptions.

.. _grid-naming:

Grid naming
-----------

The naming of grids broadly follows this specification::

    {sps_model}-{sps_version}-{sps_variant}_{imf_type}-{mass_boundaries}-{slopes}_{photoionisation_code}-{photoionisation_code_version}-{photoionisation_parameters} 

Though some of these (such as ``stellar_library``, ``slopes``, ``photoionisation_parameters``) are situation specific. For example::

    bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01 

specifies that the grid is constructed using v2.2.1 of the `Binary Population and Spectral Synthesis <https://bpass.auckland.ac.nz/>`_ (BPASS) SPS model for the binary (bin) variant. This grid assumes the Chabrier (2003) IMF between 0.1 and 300 Msol. Photoionisation modelling is performed using v23.01 of the `cloudy <https://gitlab.nublado.org/cloudy/cloudy>`_ photoionisation code assuming our `default assumptions <https://github.com/synthesizer-project/grid-generation/blob/main/src/synthesizer_grids/cloudy/params/c23.01-sps.yaml>`_. Certain SPS models also use multiple stellar spectral libraries, which we bring under sps_variant as well.
In addition to the naming, all grid files contain a complete summary of their model and photoionisation properties in attributes.


Initial Mass Function
---------------------

Grids are constructed using various initial mass functions (IMFs), often depending on the availability in the specific SPS model.
In most cases we recommend using the Chabrier (2003) IMF since this is available for most SPS models, allowing a like-for-like comparison.
If you're interested in exploring the systematic impact of changing the IMF, broken power law (bpl) IMFs may be suitable. These are named e.g. ::

    {imf_type}-{mass_boundaries}-{slopes}

e.g. for a Salpeter (1955) IMF (slope=2.35) between 0.1 and 100 Msol we would have ::

    salpeter-0.1,100-2.35

A more complex IMF, for example with two power-laws (2.0, 2.35) separated at 1 Msol, would have ::

    bpl-0.1,1.0,100-2.0,2.35

If an IMF you need is missing, please let us know by raising a feature request through an `issue <https://github.com/synthesizer-project/synthesizer/issues/new?assignees=&labels=&projects=&template=feature_request.md&title=>`_.


Photoionisation modelling
-------------------------
All the photoionisation modelling in synthesizer currently uses the `cloudy <https://gitlab.nublado.org/cloudy/cloudy>`_ photoionisation code. It can simulate a range of ISM and ionisation conditions. Our default stellar grids make certain choices to restrict the range of assumptions. In the default stellar grids, we follow the choice in `Wilkins et al. (2020) <https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.6079W/abstract>`_ and choose a reference ionisation paramter, anchored at a stellar age and metallicity of 1 Myr and 0.01, respectively (see Section 2.2.1 in `Wilkins et al. 2020 <https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.6079W/abstract>`_ ). We also choose the hydrogen density of the nebula as 1000cm\ :sup:`-3` . And we assume the nebula is ioinsation bound and hence case-B recombination holds.

* `reference_ionisation_parameter: 0.01`
* `hydrogen_density: 1.0e+3`

Common variants
---------------

* `resolution:0.1` outputs the spectra at 10x higher resolution than the `cloudy` default. Useful for looking at various absorption line indices. 
* `ionisation_parameter:X` assumes a fixed ionisation parameter `X` for the incident spectra.


Higher-dimensionality grids
---------------------------
Most SPS grids are two-dimensional, with the dimensions being `log10(age)` and `metallicity`. However synthesizer can utilise grids with higher dimensionality e.g. including varying alpha-abundance, or photoionisation parameters (e.g. `U`).

By default, certain models (e.g., parametric stars) aren't set up to handle higher dimensionality, though this may change in a future version. 
For now, we provide the functionality to handle these grids by "collapsing" over the additional axes. 
More details on this are provided in the `grids_example <grids_example>`_ notebook.

Grid list
=========

Below are examples of the pre-computed grids available in the `Box <https://sussex.app.box.com/v/SynthesizerProductionGrids>`_.

.. collapse:: Bruzual & Charlot (2003, BC03)

    * Chabrier (2003) IMF
        - bc03-2003-padova00_chabrier03-0.1,100

.. collapse:: 2016 update of Bruzual & Charlot (2003)

    * The BaSel variant
        - Chabrier (2003) IMF
            + bc03-2016-BaSeL_chabrier03-0.1,100
            + bc03-2016-BaSeL_chabrier03-0.1,100_cloudy-c23.01-sps
    
    * The Miles variant
        - Chabrier (2003) IMF
            + bc03-2016-Miles_chabrier03-0.1,100 
            + bc03-2016-Miles_chabrier03-0.1,100_cloudy-c23.01-sps
    
    * The Stelib variant
        - Chabrier (2003) IMF
            + bc03-2016-Stelib_chabrier03-0.1,100
            + bc03-2016-Stelib_chabrier03-0.1,100_cloudy-c23.01-sps

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
        - maraston24-Tenc_0.00_kroupa-0.1,100
        - maraston24-Tenc40_kroupa-0.1,100_cloudy-c23.01-sps


Exploring Grids
===============

Once you've downloaded a grid you can get started here:

.. toctree::
   :maxdepth: 1

   grids_example


Creating your own grids
=======================

For advanced users, Synthesizer contains scripts for creating your own grids from popular SPS codes, and running these through CLOUDY.
We provide scripts for doing this in the `grid-generation` repository.
Details are provided `here <../advanced/creating_grids>`_.
You will need a working installation of Synthesizer for these scripts to work, as well as other dependencies for specific codes (e.g. `CLOUDY`, `python-FSPS`).
Please reach out to us if you have questions about the pre-computed grids or grid creation.
