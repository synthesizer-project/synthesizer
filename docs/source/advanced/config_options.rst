Configuration Options
=====================

When installing Synthesizer it's possible to enable special behaviours by defining certain environment variables.

- ``WITH_OPENMP`` controls whether the C extensions will be compiled with OpenMP threading enabled. 
This can be set to an arbitrary value to compiler with OpenMP or can be the file path to the OpenMP installation directory 
(specifically the directory containing the ``include`` and ``lib`` directories). This later option is useful for systems where OpenMP is not 
automatically detected like OSX when ``libomp`` has been installed with homebrew.
- ``ENABLE_DEBUGGING_CHECKS`` turns on debugging checks within the C extensions. These (expensive) checks are extra consistency checks to ensure the code is behaving
as expected. Turning these on will slow down the code significantly.
- ``RUTHLESS`` turns on almost all compiler warnings and converts them into errors to ensure the C code in Synthesizer is clean. Note, that this does not
include the ``--pedantic`` flag because Python and Numpy themselves do not adhere to these rules and thus do not compile using ``gcc`` and this flag.
- ``ATOMIC_TIMINGS`` turns on low level timings for expensive computations. This is required to use time related profiling scripts. 
- ``CFLAGS`` allows the user to override the flags passed to the C compiler. Simply pass your desired flags to this variable.
- ``LDFLAGS`` allows the user to override the directories used during linking.

These environment variables can either be defined globally or passed on the command line when invoking ``pip install``.
To define them globally simply export them into your environment, e.g.

.. code-block:: bash

    export WITH_OPENMP=/path/to/openmp
    export ATOMIC_TIMINGS=1

To define them only while running ``pip install`` simply included them before the call to ``pip``, e.g.

.. code-block:: bash

    WITH_OPENMP=1 ATOMIC_TIMINGS=1 pip install .

Recommended Development Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When developing functionality for Synthesizer its recommended to install with debugging checks, timings, and compiler checks. 
This will ensure any code contributed not only works and is performant but follows the Synthesizer style.
To install with these options enabled use:

.. code-block:: bash

    ENABLE_DEBUGGING_CHECKS=1 ATOMIC_TIMING=1 RUTHLESS=1 pip install .

