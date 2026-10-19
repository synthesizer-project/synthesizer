Parametric Population and Wavelength Scaling
============================================

These benchmarks show how parametric operations scale with the number of populations (e.g. a bulge and a disk, up to a galaxy resolved into 100 annuli) and with the number of wavelength elements. Each operation is profiled in two ways: with a separate ``Stars`` object per population (every operation loops over them), and with a single ``Stars`` holding every population (``Stars.from_populations``), where each operation is a single call.

All tests were run using 32 threads on a COSMA8 node (two AMD EPYC 7763 processors).

Population Scaling
------------------

The following plots show how the runtime of individual operations scales with the number of populations (from 1 to 100). These tests were run using a grid with 9244 wavelength elements.

**Construction**

Building the populations from SFH and metallicity distribution functions, from binned SFZH arrays (``Stars.from_sfzh``), and combining them into a single ``Stars``.

.. image:: https://raw.githubusercontent.com/synthesizer-project/synventory/main/profiling/parametric/parametric_npops_performance_construction_nlam9244_nt32.png
   :width: 75%
   :align: center

**Spectra Generation**

The integrated emission of a combined ``Stars`` is a single extraction of every population. The emission of each population (a ``per_particle`` model) is a single matrix product with the grid.

.. image:: https://raw.githubusercontent.com/synthesizer-project/synventory/main/profiling/parametric/parametric_npops_performance_spectra_nlam9244_nt32.png
   :width: 75%
   :align: center

**Photometry**

.. image:: https://raw.githubusercontent.com/synthesizer-project/synventory/main/profiling/parametric/parametric_npops_performance_photometry_nlam9244_nt32.png
   :width: 75%
   :align: center

**Imaging**

Smoothed images at two pixel resolutions (100×100 and 1000×1000). The populations each have their own Sersic morphology, which is integrated over each pixel.

.. image:: https://raw.githubusercontent.com/synthesizer-project/synventory/main/profiling/parametric/parametric_npops_performance_imaging_nlam9244_nt32.png
   :width: 75%
   :align: center

**Memory Footprint**

These plots show the memory size of the generated spectra and photometry objects as a function of the number of populations.

.. image:: https://raw.githubusercontent.com/synthesizer-project/synventory/main/profiling/parametric/parametric_npops_performance_memory_spectra_nlam9244_nt32.png
   :width: 75%
   :align: center

.. image:: https://raw.githubusercontent.com/synthesizer-project/synventory/main/profiling/parametric/parametric_npops_performance_memory_photometry_nlam9244_nt32.png
   :width: 75%
   :align: center

Wavelength Scaling
------------------

The following plots show the scaling of spectra generation with the number of wavelength elements in the SPS grid (from 100 to 100,000 elements). These tests were run using 10 populations.

**Runtime Scaling**

.. image:: https://raw.githubusercontent.com/synthesizer-project/synventory/main/profiling/parametric/parametric_wavelength_performance_spectra_npops10_nt32.png
   :width: 75%
   :align: center

**Memory Footprint**

.. image:: https://raw.githubusercontent.com/synthesizer-project/synventory/main/profiling/parametric/parametric_wavelength_performance_memory_spectra_npops10_nt32.png
   :width: 75%
   :align: center
