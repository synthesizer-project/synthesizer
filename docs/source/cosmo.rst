Cosmological Simulations
************************

Synthesizer can be used to generate spectra from a range of different cosmological simulations. 
Below we demonstrate this for a particle based hydro code, as well as a semi-analytic model with a binned star formation and metal enrichment history.

Galaxies in a particle based simulation
=======================================

Download data
-------------

First we need some example simulation data. The ``download_camels.sh`` script will download data from the `CAMELS simulations <https://www.camel-simulations.org/>`_, a series of small volume simulations using the Simba and Illustris simulation codes.

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

The first argument here gives the grid name we wish to use, and the second the grid directory. This then creates a ``synthesizer.SpectralGrid`` object. 

We then need to load our galaxy data. There are custom data loading script for different simulation codes in ``synthesizer.load_data``. For CAMELS-Simba there is the ``load_CAMELS_SIMBA`` method::

   gals = load_CAMELS_SIMBA('data/', snap='033')


this creates ``gals``, which is a list containing a ``synthesizer.Galaxy`` object for each structure in the subfind file. These ``Galaxy`` objects contain lots of useful methods for acting on galaxies, one of which is to generate the intrinsic spectrum,::

    _g = gals[0]
    _spec = _g.integrated_stellar_spectrum(_grid)

Here we grab a single galaxy, and call ``integrated_stellar_spectrum`` providing our grid object as the first argument. This returns the spectra of the galaxy as an array, ``_spec``. 

The original wavelength is contained in the grid object - together we can use these to plot the spectrum::

   plt.loglog(_grid.lam, _spec)
   plt.show()

.. image:: images/camels_single_spec.png

The Sed object
--------------

We can also specify that ``integrated_stellar_spectrum`` returns an `Sed` object,::

   _spec = _g.integrated_stellar_spectrum(_grid, sed_object=True)

To access the luminosity and wavelength for ``_spec`` we can now do::
   
   (_spec.lam, _spec.lnu)

Why might you want to create an ``Sed`` object? This class contains a lot of useful functionality for working with SED's. For example, we can calculate the broadband luminosities

.. code-block:: python

   # first get rest frame 'flux'
   _spec.get_fnu0()

   # define a filter collection object (UVJ default)
   fc = UVJ(new_lam=_grid.lam)

   _UVJ = _spec.get_broadband_fluxes(fc)
   print(_UVJ)

.. code-block:: console

   {'U': unyt_quantity(1.07005258e+13, 'nJy'), 'V': unyt_quantity(2.28745444e+13, 'nJy'), 'J': unyt_quantity(3.34422205e+13, 'nJy')}

Generating multiple spectra
---------------------------

If we want to create spectra for multiple galaxies we can use a list comprehension::

   _specs = np.vstack([_g.integrated_stellar_spectrum(_grid) for _g in gals[:10]])

   plt.loglog(_grid.lam, _specs.T)
   plt.show()

.. image:: images/camels_multiple_spec.png

Importantly here, we don't create an SED object for each galaxy spectra. We instead create the 2D array of spectra, and then create an ``Sed`` object for the whole collection::

   # first filter by stellar mass
   mstar = np.log10(np.array([np.sum(_g.stars.masses) for _g in gals]) * 1e10)
   mask = np.where(mstar > 8)[0]

   _specs = np.vstack([gals[_g].integrated_stellar_spectrum(_grid)
                       for _g in mask])

   _specs = Sed(lam=_grid.lam, lnu=_specs)

We can then use the ``Sed`` methods on the whole collection. This is much faster than calling the method for each spectra individually, since we can take advantage of vectorisation. For example, we can calculate UVJ colours of all the selected galaxies in just a couple of lines::

   _specs.get_fnu0()
   _UVJ = _specs.get_broadband_fluxes(fc)

   UV = _UVJ['U'] / _UVJ['V']
   VJ = _UVJ['V'] / _UVJ['J']

   plt.scatter(VJ, UV, c=mstar[mask], s=4)
   plt.xlabel('VJ')
   plt.ylabel('UV')
   plt.show()

.. image:: images/camels_UVJ.png


Galaxies in a semi-analytic model
=================================

TODO

