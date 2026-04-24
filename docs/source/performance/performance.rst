Performance 
=========== 

To ensure Synthesizer is performant enough to handle the large dynamic range of possible input datasets we have put a lot of effort into optimising the codebase. 
Needless to say, we are always looking for ways to improve performance further, so if you have any suggestions or find any bottlenecks, please do not hesitate to open an issue on GitHub.

Optimisations
~~~~~~~~~~~~~

We have implemented a number of performance optimisations, including:

- Using C++ extensions for computationally intensive tasks.
- Using OpenMP for shared memory parallelism to avoid the GIL bottleneck in Python. 
- Reducing memory allocations and copies as much as possible (including removing copies inherent during ``unyt`` conversion operations). 

Profiling Suite
~~~~~~~~~~~~~~~

.. note::
    Before running the profiling suite, you will need to download the grids. See the `Downloading Grids <../getting_started/downloading_grids>`_ documentation for details.

To ensure the code remains performant, we maintain a comprehensive profiling suite to test the performance of the codebase. 
The profiling scripts and documentation can be found in the `profiling directory <https://github.com/synthesizer-project/synthesizer/tree/main/profiling>`_ of the repository.

**Profiling Categories:**

- **Particle and Wavelength Scaling**: How individual operations scale with problem size (number of particles or wavelength elements)
- **Pipeline Profiling**: Real-world benchmarks with multiple operations performed in sequence
- **Strong Scaling**: How performance scales with thread count for fixed problem sizes

The profiling suite includes scripts to:

- Run timing benchmarks for various operations and configurations
- Profile memory usage with configurable sampling frequencies  
- Analyse and visualise profiling results
- Generate the performance plots shown in this documentation

See the `profiling README <https://github.com/synthesizer-project/synthesizer/tree/main/profiling>`_ for details on running the profiling suite and reproducing these benchmarks.

Hardware Specifications
~~~~~~~~~~~~~~~~~~~~~~~

The benchmarks shown in this documentation were run on the Cosma8 HPC at Durham University. The output of the ``lscpu`` command is shown below, which gives an idea of the hardware used for these tests:

.. code-block:: 

    Architecture:             x86_64
      CPU op-mode(s):         32-bit, 64-bit
      Address sizes:          43 bits physical, 48 bits virtual
      Byte Order:             Little Endian
    CPU(s):                   128
      On-line CPU(s) list:    0-127
    Vendor ID:                AuthenticAMD
      Model name:             AMD EPYC 7542 32-Core Processor
        CPU family:           23
        Model:                49
        Thread(s) per core:   2
        Core(s) per socket:   32
        Socket(s):            2

Most benchmarks were run using 8 threads unless otherwise specified.

Memory Footprint Note
~~~~~~~~~~~~~~~~~~~~~

.. note::
   The memory plots in this documentation measure different aspects of memory usage depending on the benchmark:
   
   - **Pipeline profiling**: Shows RSS (Resident Set Size) memory sampled at high frequency during execution. This captures the total memory footprint including transient spikes.
   - **Individual operation benchmarks**: Show the size of the final objects stored in memory (e.g., the generated spectra or photometry data). While these represent the permanent memory cost added to your session, there may be transient spikes in memory usage during the actual computation that are slightly higher than these values.

Performance Benchmarks
~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   particle_wavelength_scaling
   pipeline_profiling
   strong_scaling
