Parallelism
===========

Synthesizer supports both shared memory and distributed memory parallelism, giving users fine-grained control over how to parallelise their workflows.

- **Shared Memory (OpenMP)**: Built-in threading for computationally intensive operations within individual galaxies
- **Distributed Memory (MPI)**: Partitioning of galaxy populations across processors left in the hands of user, with interfaces for the Comm object and rank/size information

This hybrid approach allows you to avoid parallelism overheads where not needed, while enabling efficient scaling for large galaxy populations.

Shared Memory Parallelism (OpenMP)
----------------------------------

To use OpenMP threading in Synthesizer, you need to first have OpenMP installed or an OpenMP compliant compiler on your system, and to compile the code with the appropriate flags.
For more details see the `installation instructions <../getting_started/installation.rst>`_ and the `configuration options <../advanced/config_options.rst>`_.

> Note: The Clang compiler on macOS does not support OpenMP, so you will either need to install OpenMP via Homebrew or use a different compiler such as GCC. 

Testing OpenMP
^^^^^^^^^^^^^^

To check that OpenMP is indeed being used in your code, you can import and use the ``check_openmp`` function.

.. code-block:: python

    from synthesizer import check_openmp
    check_openmp()

If OpenMP has been successfully configured, this function will return ``True``, otherwise it will return ``False``.

Using OpenMP Threading
^^^^^^^^^^^^^^^^^^^^^^

To avoid the pitfalls of the Python Global Interpreter Lock (GIL), we have focused on parallelisation of the C++ extensions used for spectra generation,
integration, Line-Of-Sight (LOS) surface density calculations, imaging, and other computationally intensive tasks. 

Making use of these threadpools is as simple as passing the ``nthreads`` argument to the relevant function. For example, to use 4 threads when generating spectra:

.. code-block:: python

    galaxy.get_spectra(..., nthreads=4)

The exact same would be true for any other function that supports OpenMP threading.

.. code-block:: python

    galaxy.stars.get_los_column_density(..., nthreads=4)

Distributed Memory Parallelism (Pipeline)
------------------------------------------

For processing large galaxy populations, Synthesizer's ``Pipeline`` object provides a framework for distributed parallelism, but **leaves the partitioning in your hands**. 

**Why User-Controlled Partitioning?**

Different scientific workflows have different parallelisation strategies:

- Simple embarrassingly parallel splits across galaxy populations
- Load-balanced partitioning based on particle counts
- Spatially-aware splits for simulation snapshots
- Custom groupings based on galaxy properties

By leaving partitioning to the user, you can choose the optimal strategy for your specific use case.

**Example: MPI with Pipeline**

Here's a simple example of using MPI to process different galaxy subsets on different ranks:

.. code-block:: python

    from mpi4py import MPI
    from synthesizer import Pipeline
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Load all galaxies (or use lazy loading for large datasets)
    all_galaxies = load_galaxies()
    
    # Partition galaxies across ranks (simple split)
    my_galaxies = all_galaxies[rank::size]
    
    # Each rank processes its subset
    pipeline = Pipeline(emission_model=model, nthreads=8, comm=comm)
    pipeline.add_galaxies(my_galaxies)
    # ... configure pipeline operations ...
    pipeline.run()
    
    # Save results (each rank writes its own file or gather results)
    save_results(pipeline.galaxies, rank=rank)

**Load Balancing**

For better load balancing, you can partition based on computational cost (e.g., particle counts):

.. code-block:: python

    # Sort galaxies by particle count
    galaxies_sorted = sorted(all_galaxies, 
                            key=lambda g: len(g.stars.masses),
                            reverse=True)
    
    # Round-robin assignment for load balancing
    my_galaxies = galaxies_sorted[rank::size]

**Combining MPI and OpenMP**

For optimal performance on HPC systems, combine distributed (MPI) and shared memory (OpenMP) parallelism:

.. code-block:: python

    # Use MPI for galaxy-level parallelism
    my_galaxies = all_galaxies[rank::size]
    
    # Use OpenMP threading within each galaxy
    pipeline = Pipeline(emission_model=model, nthreads=8, comm=comm)
    pipeline.add_galaxies(my_galaxies)
    pipeline.run()

This hybrid approach allows you to scale to thousands of cores on HPC systems while maintaining efficiency.

