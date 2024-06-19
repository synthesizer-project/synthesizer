Sed
*****

Introduction
============
A Spectral Energy Distribution or SED describes the energy distribution of any emitting body as a function of frequency/wavelength. In `synthesizer`, spectra when generated are stored in `Sed` object. One can take spectra directly from a `Grid <../grids/grids.rst>`_,  computed from a `Galaxy <../galaxies/galaxies.rst>`_ or a Galaxy `Component <../components/components.rst>`_ the resulting calculated spectra are stored in `Sed` objects. `SED` object in synthesizer is generally agnostic of where the input spectra is coming from, and thus can be inititalised as a function of any arbitrary frequency/wavelength and the corresponding flux/luminosity density. The `Sed` object has the ability to contain multiple spectra (multiple galaxies or particles).

The `Sed` object contains the necessary functions to calculate broadband luminosities or fluxes (wavelength windows or on `filters <../filters/filters.rst>`_), different spectral indices (e.g. balmer break, UV-continuum slope) or apply attenuation by dust.

.. toctree::
   :maxdepth: 1

   sed_example
