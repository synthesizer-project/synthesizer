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
    pip install .

If you want to install the code in *editable* mode, so any changes in the code base will be reflected in the installation::

    pip install .

Make sure you stay up to date with the latest versions through git::

    cd synthesizer
    git pull origin main

Installing with optional flags
##############################

DISCLAIMER: most users do not need to worry about this section. Synthesizer, by design, simplifies compilation so the user doesn't need to think about it.

Synthesizer uses C extensions for much of the heavy lifting done in the background to derive spectra. By default Synthesizer will use ``-std=c99 -Wall -O3 -ffast-math -g`` (on a unix-system) to optimise agressively. You can override this by modifying the compiler flags and linker arguments at the point of install, e.g.

```
CFLAGS=... LDFLAGS=... pip install .
```

Setting these environment variables will override the default flags. For example, to compile with debugging symbols and no optimisation, you could use

```
CFLAGS="-std=c99 -Wall -g" LDFLAGS="-g" pip install .
```

which would be recommended if you are developing the code, particularly the C extensions. In addition to disabling optimisation and turning on debugging symbols, you can also turn on debugging checks by adding ``WITH_DEBUGGING_CHECKS=1``, e.g.

```
CFLAGS="-std=c99 -Wall -g" WITH_DEBUGGING_CHECKS=1 pip install .
```

Downloading SPS grids
#####################

Once you've installed the code, you're almost ready to get started with Synthesizer. The last step is to download an SPS *grid* file, described in the next section.
