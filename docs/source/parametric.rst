Parametric Star Formation Histories
***********************************

Synthesizer can be used to generate spectra for simple parametric star formation histories.
Below we show some examples; you'll need the following modules::

    import numpy as np
    import matplotlib.pyplot as plt
    from unyt import Myr

    from synthesizer.grid import SpectralGrid
    from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh
    from synthesizer.parametric.galaxy import SEDGenerator
    

Making an SED
=============

We first need to load a grid file (see :doc:`grids` for details)::

   grid_name = 'bpass-v2.2.1-bin_chab-100_cloudy-v17.03_log10Uref-2'
   grid_dir = '/our/synthesizer/data/directory/synthesizer_data/grids'
   grid = SpectralGrid(grid_name, grid_dir=grid_dir)

Now define the parameters of the star formation history and its functional form::

    sfh_p = {'duration': 10 * Myr}
    stellar_mass = 1E8
    
    sfh = SFH.Constant(sfh_p)  # constant star formation

We can print some of these properties

.. code-block:: python

   sfh.summary()  # print sfh summary

.. code-block:: console

    > ----------
    > SUMMARY OF PARAMETERISED SFH
    > <class 'synthesizer.parametric.sfzh.SFH.Constant'>
    > duration: 10 Myr
    > median age: 5.00 Myr
    > mean age: 5.00 Myr


Now do the same for the metal enrichment history::
    
    Z_p = {'log10Z': -2.0}  # can also use linear metallicity e.g. {'Z': 0.01}
    Zh = ZH.deltaConstant(Z_p)  # constant metallicity

Get the 2D star formation and metal enrichment history for the given SPS grid. This is (age, Z)::

    sfzh = generate_sfzh(grid.log10ages, grid.metallicities, sfh, Zh, stellar_mass=stellar_mass)

We can now use this model to generate a galaxy SED, and plot the intrinsic spectra::

    galaxy = SEDGenerator(grid, sfzh)
    galaxy.plot_spectra()

.. image:: images/parametric_sed_A.png
    :scale: 50 %

This is assuming no dust or nebular attenuation, so the intrinsic spectra emission is identical to the total. How about we apply a simple dust and gas screen?::

    galaxy.screen(tauV = 0.1)
    galaxy.plot_spectra()

.. image:: images/parametric_sed_B.png
    :scale: 50 %

Should be identical to above::

    galaxy.pacman(tauV = 0.1)
    galaxy.plot_spectra()

.. image:: images/parametric_sed_C.png
    :scale: 50 %

We can adjust the fraction of light that experiences nebular reprocessing using the ``fesc`` keyword::
    
    galaxy.pacman(fesc = 0.5)
    galaxy.plot_spectra()

.. image:: images/parametric_sed_D.png
    :scale: 50 %

We can also explicitly adjust the fraction of Lyman-alpha escape. In this example we set it to zero::

    galaxy.pacman(fesc = 0.0, fesc_LyA = 0.0)
    galaxy.plot_spectra()

.. image:: images/parametric_sed_E.png
    :scale: 50 %

Finally, we can combine all of the above::
    
    galaxy.pacman(fesc=0.5, fesc_LyA=0.5, tauV=0.2)
    galaxy.plot_spectra()

.. image:: images/parametric_sed_F.png
    :scale: 50 %

All of these spectra are generated in the rest-frame. To shift to the observer frame we require some information on the cosmology we assume. We first load an astropy cosmology instance. We also load and assume the Madau+96 IGM absorption::

    from astropy.cosmology import Planck18 as cosmo
    from synthesizer.igm import Madau96

Let's assume this galaxy is at :math:`z = 10`::

    z = 10.  # redshift
    sed = galaxy.spectra['total']  # choose total SED
    sed.get_fnu(cosmo, z, igm=Madau96())  # generate observed frame spectra

We can also calculate broadband luminosities from this ``sed`` object. Here we choose a number of JWST NIRCam and MIRI filters (details on filter creation are provided in :doc:`filters`)::
    
    from synthesizer.filters import SVOFilterCollection

    # define a list of filter codes
    filter_codes = [f'JWST/NIRCam.{f}' for f in ['F090W', 'F115W', 'F150W',
                                                 'F200W', 'F277W', 'F356W', 'F444W']]  

    filter_codes += [f'JWST/MIRI.{f}' for f in ['F770W']]
    fc = SVOFilterCollection(filter_codes, new_lam=sed.lamz)

    # --- measure broadband fluxes
    fluxes = sed.get_broadband_fluxes(fc)

    for filter, flux in fluxes.items():
        print(f'{filter}: {flux:.2f}')  # print broadband fluxes

.. code-block:: console

   > JWST/NIRCam.F090W: 0.00 nJy
   > JWST/NIRCam.F115W: 0.00 nJy
   > JWST/NIRCam.F150W: 52.28 nJy
   > JWST/NIRCam.F200W: 47.10 nJy
   > JWST/NIRCam.F277W: 41.89 nJy
   > JWST/NIRCam.F356W: 35.17 nJy
   > JWST/NIRCam.F444W: 34.84 nJy
   > JWST/MIRI.F770W: 29.77 nJy

Make plot of observed including broadband fluxes (if filter collection object given)::

    galaxy.plot_observed_spectra(cosmo, z, fc=fc, spectra_to_plot=['total'])

.. image:: images/parametric_broadband.png
    :scale: 50 %

