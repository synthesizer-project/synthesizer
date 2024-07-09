Installation
************

To get started with ``synthesizer`` you need to complete the following setup steps.

- Create a python environment
- Clone the code base
- Install the code
- Download a `grid`

Creating a python environment
#############################

First create a virtual python environment. ``synthesizer`` has been tested with Python 3.10, 3.11 and 3.12. You should replace this path with where you wish to install your virtual environment:

.. code-block:: bash

    python3 -m venv /path_to_new_virtual_environment
    source /path_to_new_virtual_environment/bin/activate

Cloning and installing the code
###############################

You can then clone the latest version of ``synthesizer`` from `github <https://github.com/flaresimulations/synthesizer>`_, and finally install it:

.. code-block:: bash

    git clone https://github.com/flaresimulations/synthesizer.git
    cd synthesizer
    pip install .

Make sure you stay up to date with the latest versions through git:

.. code-block:: bash

    cd synthesizer
    git pull origin main

If you are an advanced user directly developing the synthesizer code, and want to install the code in *editable* mode (so any changes in the code base will be reflected in the installation) please add the following flag:

.. code-block:: bash

    pip install -e .

Installing with OpenMP
######################

DISCLAIMER: This section is only important if you want to make use of shared memory parallelism for large calculations.

To make use of Synthesizer's `shared memory parallelism <../parallelism/openmp.rst>`_ you must first have OpenMP installed on your system. 
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

For more details on configuration options, see the `configuration options docs <../advanced/config_options.rst>`_.

Downloading grids
#################

Once you've installed the code, you're almost ready to get started with Synthesizer. The last step is to download a *grid* file, described in the next section.
