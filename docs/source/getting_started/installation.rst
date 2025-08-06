Installation
************

This guide will walk you through creating an isolated environment and installing the package.

If you instead want to set up a new installation for development see the installation instructions in the `contributing guidelines <contributing.rst>`_.

**Note**: We do not currently support Windows, to use Synthesizer on Windows please install the Windows Subsystem for Linux (WSL).

Creating a python environment
#############################

First create a virtual python environment. Synthesizer has been tested with Python 3.10, 3.11, 3.12, and 3.13. You should replace this path with where you wish to install your virtual environment:

.. code-block:: bash

    python3 -m venv /path_to_new_virtual_environment
    source /path_to_new_virtual_environment/bin/activate

Installing from PyPI (Recommended)
##################################

To install from PyPI, simply run:

.. code-block:: bash

    pip install cosmos-synthesizer

This will install the latest stable version of Synthesizer and all its dependencies. 

    Note that we package Synthesizer with only a source distribution, so that the package's C extensions are compiled on your machine. This is to ensure the package is compiled with the correct flags for your system and thus the best optimisations.

Installing from source 
###################### 

To install the latest unstable version from source, clone the repository and install the package from the root directory: 

.. code-block:: bash

    git clone https://github.com/synthesizer-project/synthesizer.git
    cd synthesizer
    pip install .

Installing with Optional Configuration Options
##############################################

Synthesizer provides several optional configuration options that can be set at installation time. These options can be set by setting environment variables prior to installation. Below we detail the most important of these options.

For a full list of configuration options and their possible values see the `configuration options docs <../advanced/config_options.rst>`_.

Installing with OpenMP support 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DISCLAIMER: This section is only important if you want to make use of shared memory parallelism for large calculations.

To make use of synthesizer's `shared memory parallelism <../performance/openmp.rst>`_ you must first have OpenMP installed on your system.
Most compilers come with OpenMP baked in with a few exceptions. 
This means installation with OpenMP is as simple as setting a flag at installation:

.. code-block:: bash

    WITH_OPENMP=1 pip install .

On Linux this approach should be sufficient in almost all cases. 

On OSX OpenMP must be installed via `homebrew <https://brew.sh/>`_ and ``setuptools`` appears to struggle automatically finding the install location.
In this situation, and any others where the automatic locating of OpenMP fails, the path to the installation can be stated explicitly:

.. code-block:: bash

    WITH_OPENMP=/opt/homebrew/Cellar/libomp/18.1.6/ pip install .

Note that the path should point to the directory containing the ``include`` and ``lib`` directories.

Optional Dependencies
##################### 

Synthesizer provides several optional dependency groups to cater to different use cases. These groups can be installed directly from PyPI using the following syntax:

.. code-block:: bash

    pip install cosmos-synthesizer[<group>] 

Or when installing from source: 

.. code-block:: bash

    pip install .[<group>]

The available groups are:

- **Development** (``dev``): Tools to help developing including linting and formatting.
- **Testing** (``test``): Frameworks and utilities for running tests.
- **Documentation** (``docs``): Packages required to build the project documentation.
- **Simulation-specific loaders**: Additional libraries required for loading certain simulation data:
  - ``bluetides``: For working with Bluetides simulation files. 
  - ``eagle``: For working with Eagle simulation files.

For example, to install with development dependencies, run:

.. code-block:: bash

    pip install cosmos-synthesizer[dev]

Multiple optional dependency groups can be installed in one command. For instance, to install both the testing and documentation dependencies, run:

.. code-block:: bash

    pip install cosmos-synthesizer[test,docs]

Initialising Synthesizer
########################

Synthesizer has a small number of data files and directories it needs to function correctly. 
In most circumstances you don't need to worry about these and everything will be default with automatically.
The first time you import Synthesizer it will automatically create this directory and tell you where it is and what files have been placed there. 

However, this can be invoked manually using the ``synthesizer-init`` command. This command includes some extra options, here is the output of ``synthesizer-init --help``:

.. code-block:: bash

    usage: synthesizer-init [-h] [--force] [--print]

    Initialise the Synthesizer data directory.

    options:
      -h, --help   show this help message and exit
      --force, -f  Force re-initialisation even if directories already exist.
      --print, -p  Print a report showing the paths and file locations.

Invoking ``synthesizer-init`` alone will run the initialisation process or exit silently if the directories already exist. To see a report of the paths and file locations, you can use the ``--print`` or ``-p`` option.

To clear out the Synthesizer data directory and re-initialise it, you can use the ``--force`` or ``-f`` option. This will remove any existing directories and files before re-initialising. 

To simply clear the Synthesizer data directory without re-initialising, you can run the ``synthesizer-clear`` command. This command takes no arguments.

Environment Variables 
##################### 

If you want to customise any of these locations you can define a set of environment varaibles to do so. 

The environment variables are:
- ``SYNTHESIZER_DIR``: The base directory for Synthesizer files. 
- ``SYNTHESIZER_DATA_DIR``: The directory for Synthesizer data files. 
- ``SYNTHESIZER_INSTRUMENT_CACHE``: The directory for Synthesizer's premade instrument files.
- ``SYNTHESIZER_TEST_DATA_DIR``: The directory for Synthesizer's test data files.

Setting any of these environment variables will result in their creation and population the next time you run ``synthesizer-init`` or import Synthesizer.
