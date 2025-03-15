Configuration Options
=====================

Synthesizer uses C extensions for much of the heavy lifting done in the background. When installing synthesizer it's possible to enable special behaviours and control the compilation of these C extensions by defining certain environment variables.

- ``WITH_OPENMP`` controls whether the C extensions will be compiled with OpenMP threading enabled. This can be set to an arbitrary value to compiler with OpenMP or can be the file path to the OpenMP installation directory (specifically the directory containing the ``include`` and ``lib`` directories). This later option is useful for systems where OpenMP is not automatically detected like OSX when ``libomp`` has been installed with homebrew.
- ``ENABLE_DEBUGGING_CHECKS`` turns on debugging checks within the C extensions. These (expensive) checks are extra consistency checks to ensure the code is behaving as expected. Turning these on will slow down the code significantly.
- ``RUTHLESS`` turns on almost all compiler warnings and converts them into errors to ensure the C code in synthesizer is clean. Note, that this does not include the ``--pedantic`` flag because Python and Numpy themselves do not adhere to these rules and thus do not compile using ``gcc`` and this flag.
- ``ATOMIC_TIMING`` turns on low level timings for expensive computations. This is required to use time related profiling scripts. 
- ``CFLAGS`` allows the user to override the flags passed to the C compiler. Simply pass your desired flags to this variable.
- ``LDFLAGS`` allows the user to override the directories used during linking.
- ``EXTRA_INCLUDES`` allows the user to provide any extra include directories that fail to be automatically picked up.

These environment variables can either be defined globally or passed on the command line when invoking ``pip install``.
To define them globally simply export them into your environment, e.g.

.. code-block:: bash

    export WITH_OPENMP=/path/to/openmp
    export ATOMIC_TIMING=1

To define them only while running ``pip install`` simply included them before the call to ``pip``, e.g.

.. code-block:: bash

    WITH_OPENMP=1 ATOMIC_TIMING=1 pip install cosmos-synthesizer

These options can be used both when installing from source and from PyPI, e.g.

.. code-block:: bash

    WITH_OPENMP=1 pip install cosmos-synthesizer

and 

.. code-block:: bash

    ENABLE_DEBUGGING_CHECKS=1 pip install .

Will both work as expected.

Changing Compiler Flags
^^^^^^^^^^^^^^^^^^^^^^^

By default synthesizer will use ``CFLAGS=-std=c99 -Wall -O3 -ffast-math -g`` (on a unix-system) to optimise agressively. To modify these flags just define ``CFLAGS`` when installing, e.g.

.. code-block:: bash

    CFLAGS=... LDFLAGS=... pip install .


Setting these environment variables will override the default flags. Note that synthesizer will **not** santise any requested flags and a poor choice could result in a failed compilation or poor performance.

The build process will generate a log file (`build_synth.log`) which details the compilation process and any choices that were made about the requested flags. If you encounter any issues, please check this log file for more information.


Recommended Development Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When developing functionality for synthesizer its recommended to install with debugging checks, timings, and compiler checks.
This will ensure any code contributed not only works and is performant but adheres to best coding practices.
To install with these options enabled use:

.. code-block:: bash

    ENABLE_DEBUGGING_CHECKS=1 ATOMIC_TIMING=1 RUTHLESS=1 pip install .

Debugging
^^^^^^^^^

For debugging specifically you should also compile with debugging symbols and no optimisation, e.g.

.. code-block:: bash
    
    CFLAGS="-std=c99 -Wall -g" LDFLAGS="-g" ENABLE_DEBUGGING_CHECKS=1 pip install .

However, the lack of optimisation with the inclusion of debugging checks, while necessary to debug, will slow the code down extensively.

Configuration Options and ``pip``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When reinstalling from pip and changing configuration options it's important to uninstall the previous version first and to ignore any cached versions of the package. This is because the installation needs to recompile the C extensions from source. 

For instance, to reinstall with OpenMP support:

.. code-block:: bash

    pip uninstall cosmos-synthesizer
    WITH_OPENMP=1 pip install --no-cache-dir cosmos-synthesizer

When installing from source, the package will always be recompiled from source and thus the configuration options will be applied.





