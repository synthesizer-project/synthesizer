Generating Spectra
******************

We provide a number of example scripts in the :code:`example` directory that demonstrate how to run synthesizer to generate spectra for various use cases.


Example 1: galaxies in a cosmological simulation
================================================

Download data
-------------

First we need some example simulation data. The ``download_camels.sh`` script will download data from the `CAMELS simulations https://www.camel-simulations.org/`_, a series of small volume simulations using the Simba and Illustris simulation codes.

``./download_camels.sh data/ 033``

This script downloads the snapshot and subhalo data for a given snapshot (*033*) that we need to generate spectra for each galaxy in that snapshot.

Generating spectra
------------------

The script ``camels.py`` contains an example of loading this data and generating a spectra for each galaxy.

First we load some modules::
    
    import numpy as np
    import matplotlib.pyplot as plt

    from synthesizer import grid
    from synthesizer.load_data import load_CAMELS_SIMBA

We then need to load a *grid file*, which we downloaded previously (see grid creation)::

    _grid = grid.SpectralGrid('bc03_chabrier03', grid_dir=f'../../synthesizer_data/grids/bc03_chabrier03')

The first argument here gives the grid name we wish to use, and the second the grid directory. If no grid directory is specified the method will look within a default location in the package directory, ``data/grids``. This then creates a ``synthesizer.SpectralGrid`` object, which derives from the more general ``synthesizer.Grid`` class. 

We then need to load our galaxy data. There are custom data loading script for different simulation codes in ``synthesizer.load_data``. For CAMELS-Simba there is the ``load_CAMELS_SIMBA`` method::

   gals = load_CAMELS_SIMBA('data/', snap='033')


this creates ``gals``, which is a list containing a ``synthesizer.Galaxy`` object for each structure in the subfind file. These ``Galaxy`` objects contain lots of useful methods for acting on galaxies, one of which is to generate the intrinsic spectrum,::

    _g = gals[0]
    _spec = _g.integrated_stellar_spectrum(_grid)

Here we grab a single galaxy, and call ``integrated_stellar_spectrum`` providing our grid object as the first argument.

This returns the spectra of the galaxy as an array, ``_spec``. The original wavelength is contained in the grid object - together we can use these to plot the spectrum::

   plt.loglog(_grid.lam, _spec)
   plt.show()

If we want to create spectra for multiple galaxies we can use a list comprehension::

   _specs = np.vstack([_g.integrated_stellar_spectrum(_grid) for _g in gals[:10]])

   plt.loglog(_grid.lam, _specs.T)
   plt.show()

