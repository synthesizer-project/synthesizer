Installation
************

First create a virtual python environment (``synthesizer`` has been tested with Python 3.10). You should replace this path with where you wish to install your virtual environment::

    python3 -m venv /path/to/new/virtual/environment
    source /path/to/new/virtual/environment/bin/activate

You can then clone the latest version of ``synthesizer`` from github, install the requirements, and finally install ``synthesizer``::

    git clone https://github.com/flaresimulations/synthesizer.git
    cd synthesizer
    pip install -r requirements.txt
    python setup.py install


