Grids
*****

Introduction
============

Most of the functionality of Synthesizer is reliant on *grid files*. These are typically precomputed multi-dimensional arrays of spectra (and lines) from Stellar Population Synthesis (SPS) models for a range of ages and metallicities, and potentially other parameters (see below).
Grids can also represent the emission from other sources, e.g. active galactic nuclei.

There is a low-resolution test grid available via the ``synthesizer-download`` command line tool, but for actual projects you will need to download one or more full production grids from `Box <https://sussex.box.com/v/SynthesizerProductionGrids>`_. See details below on where on your system to download these grids and how to load them. 


.. toctree::
   :maxdepth: 1

   pre-computed_grids
   grids_example
   grids_modify
   grids_lines


