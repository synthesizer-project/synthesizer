MPI Scaling
===========

The Pipeline can distribute galaxies across MPI ranks, with each rank processing its own galaxies using OpenMP threads (see :doc:`../pipeline/pipeline_example`). These tests measure how the Pipeline scales as ranks are added, running the same operations as the :doc:`pipeline_profiling` benchmarks (LOS optical depths, SFZH/SFH, spectra, photometry, lines and imaging, in both the rest and observer frame) on synthetic galaxies.

Each rank uses 16 threads, pinned to one NUMA domain, with 8 ranks per COSMA8 node (two AMD EPYC 7763 processors, 8 NUMA domains of 16 cores), from 1 rank up to 32 ranks across 4 nodes. Only ``Pipeline.run`` is timed, and the time shown is that of the slowest rank, since that is how long the run takes.

Every test is run with two galaxy populations:

- **Balanced**: every galaxy has 10,000 star particles (with 2,000 gas particles and 200 black holes). The galaxy properties (masses, ages, metallicities, positions and so on) are randomly seeded on each rank, but each galaxy costs about the same, so the work per rank is balanced by construction. These runs show the scaling the Pipeline can achieve in the ideal case.
- **Power law**: the number of star particles in each galaxy is drawn from :math:`dN/dn \propto n^{-2}` between :math:`10^3` and :math:`10^5` (with gas and black holes in the same proportions), so the costs of individual galaxies span two orders of magnitude, as in real galaxy populations. The galaxies are partitioned across the ranks by particle count, assigning the largest remaining galaxy to the least loaded rank, and galaxies with more than 10,000 star particles are processed in chunks (``max_npart=10000``). These runs show the scaling on a more realistic workload.

Weak Scaling
------------

Every rank processes 100 galaxies, so the total work grows with the number of ranks. Ideally the run time stays constant, and the parallel efficiency is :math:`T(1) / T(n)`.

Balanced
^^^^^^^^

.. code-block:: bash

    bash profiling/mpi/run_mpi_scaling.sh --mode weak --ngalaxies 100

.. image:: https://raw.githubusercontent.com/synthesizer-project/synventory/main/profiling/mpi_weak/mpi_weak_scaling.png
    :width: 100%
    :align: center

Power law
^^^^^^^^^

.. code-block:: bash

    bash profiling/mpi/run_mpi_scaling.sh --mode weak --ngalaxies 100 \
        --particle-dist powerlaw --max-npart 10000

.. image:: https://raw.githubusercontent.com/synthesizer-project/synventory/main/profiling/mpi_weak/mpi_weak_powerlaw_scaling.png
    :width: 100%
    :align: center

Strong Scaling
--------------

1,000 galaxies in total are split across the ranks. Ideally the run time halves each time the number of ranks doubles, and the parallel efficiency is :math:`T(1) / (n\,T(n))`. Once there are few galaxies per rank, any imbalance in the work per rank shows up directly, since the run waits for the slowest rank. The power law catalogue is drawn from the same seed at every rank count, so every point processes the same galaxies.

Balanced
^^^^^^^^

.. code-block:: bash

    bash profiling/mpi/run_mpi_scaling.sh --mode strong --ngalaxies 1000

.. image:: https://raw.githubusercontent.com/synthesizer-project/synventory/main/profiling/mpi_strong/mpi_strong_scaling.png
    :width: 100%
    :align: center

The run time falls from 1943 s on 1 rank to 62.5 s on 32 ranks, a speedup of 31.1 and a strong scaling efficiency of 97%.

Power law
^^^^^^^^^

.. code-block:: bash

    bash profiling/mpi/run_mpi_scaling.sh --mode strong --ngalaxies 1000 \
        --particle-dist powerlaw --max-npart 10000

.. image:: https://raw.githubusercontent.com/synthesizer-project/synventory/main/profiling/mpi_strong/mpi_strong_powerlaw_scaling.png
    :width: 100%
    :align: center

Reproducing these tests
-----------------------

The scripts are in the `profiling/mpi directory <https://github.com/synthesizer-project/synthesizer/tree/main/profiling/mpi>`_. They need an MPI implementation and ``mpi4py``, and should be run inside a job allocation large enough for the largest rank count.
