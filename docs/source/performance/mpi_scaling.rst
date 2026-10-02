MPI Scaling
===========

The Pipeline can distribute galaxies across MPI ranks, with each rank processing its own galaxies using OpenMP threads (see :doc:`../pipeline/pipeline_example`). These tests measure how the Pipeline scales as ranks are added, running the same operations as the :doc:`pipeline_profiling` benchmarks (LOS optical depths, SFZH/SFH, spectra, photometry, lines and imaging, in both the rest and observer frame) on fake galaxies of 10,000 star particles.

The galaxies are randomly seeded on each rank, so their properties (masses, ages, metallicities, positions and so on) differ between ranks. Every galaxy has the same number of particles though (10,000 stars, 2,000 gas particles and 200 black holes), so each galaxy costs about the same and the work per rank is inherently balanced. Real galaxy populations span orders of magnitude in particle count, so on real data the scaling also depends on how evenly galaxies are distributed across ranks.

Each rank uses 16 threads, pinned to one NUMA domain, with 8 ranks per COSMA8 node (two AMD EPYC 7763 processors, 8 NUMA domains of 16 cores), from 1 rank up to 32 ranks across 4 nodes. Only ``Pipeline.run`` is timed, and the time shown is that of the slowest rank, since that is how long the run takes.

Weak Scaling
------------

Every rank processes 25 galaxies, so the total work grows with the number of ranks. Ideally the run time stays constant, and the parallel efficiency is :math:`T(1) / T(n)`.

.. code-block:: bash

    bash profiling/mpi/run_mpi_scaling.sh --mode weak --ngalaxies 25

.. image:: https://raw.githubusercontent.com/synthesizer-project/synventory/main/profiling/mpi_weak/mpi_weak_scaling.png
    :width: 100%
    :align: center

Going from 1 rank (25 galaxies) to 32 ranks over 4 nodes (800 galaxies), the run time grows from 48.6 s to 49.8 s, a weak scaling efficiency of 98%.

Strong Scaling
--------------

1,000 galaxies in total are split as evenly as possible across the ranks. Ideally the run time halves each time the number of ranks doubles, and the parallel efficiency is :math:`T(1) / (n\,T(n))`. Once there are few galaxies per rank, any imbalance in the work per rank shows up directly, since the run waits for the slowest rank.

.. code-block:: bash

    bash profiling/mpi/run_mpi_scaling.sh --mode strong --ngalaxies 1000

.. image:: https://raw.githubusercontent.com/synthesizer-project/synventory/main/profiling/mpi_strong/mpi_strong_scaling.png
    :width: 100%
    :align: center

The run time falls from 1943 s on 1 rank to 62.5 s on 32 ranks, a speedup of 31.1 and a strong scaling efficiency of 97%.

Reproducing these tests
-----------------------

The scripts are in the `profiling/mpi directory <https://github.com/synthesizer-project/synthesizer/tree/main/profiling/mpi>`_. They need an MPI implementation and ``mpi4py``, and should be run inside a job allocation large enough for the largest rank count.
