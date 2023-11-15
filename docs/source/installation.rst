Installation
************

To get started with ``synthesizer`` you need to complete the following setup steps.

- Create a python environment
- Clone the code base
- Install the code
- Download a stellar population synthesis (SPS) grid

Creating a python environment
#############################

First create a virtual python environment (``synthesizer`` has been tested with Python 3.10). You should replace this path with where you wish to install your virtual environment::

    python3 -m venv /path_to_new_virtual_environment
    source /path_to_new_virtual_environment/bin/activate

Cloning and installing the code
###############################

You can then clone the latest version of ``synthesizer`` from `github <https://github.com/flaresimulations/synthesizer>`_, and finally install it::

    git clone https://github.com/flaresimulations/synthesizer.git
    cd synthesizer
    pip install -e .

This installs it in *editable* mode, so any changes in the code base will be reflected in the installation. 

Make sure you stay up to date with the latest versions through git::

    cd synthesizer
    git pull origin main

Downloading SPS grids
#####################

Once you've installed the code, you're almost ready to get started with Synthesizer. The last step is to download an SPS *grid* file, described in the next section.
