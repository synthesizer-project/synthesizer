Sed
*****

A Spectral Energy Distribution, or SED, describes the energy distribution of an emitting body as a function of frequency / wavelength.
In synthesizer, generated spectra are stored in ``Sed`` objects.

There are a number of different ways to generate spectra in synthesizer, but in every case the resulting SED is always stored in an ``Sed`` object.
An ``Sed`` object in synthesizer is generally agnostic of where the input spectra comes from; they can therefore be inititalised provided any arbitrary frequency / wavelength and the corresponding flux / luminosity density.
An ``Sed`` object has the ability to contain multiple spectra (multiple galaxies or particles).

The ``Sed`` class contains a number of methods for calculating derived properties, such as broadband luminosities / fluxes (within wavelength windows or on `filters <../filters/filters.rst>`_) and spectral indices (e.g. balmer break, UV-continuum slope), and for modifying the spectra themselves, for example by applying an attenuation curve due to dust.

In the following example we introduce the ``Sed`` class, and some simple functionality.

.. toctree::
   :maxdepth: 1

   sed_example
