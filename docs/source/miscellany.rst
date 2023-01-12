Miscellany
**********

.. _filters:
Filters
=======

Synthesizer allows us to define filters quite flexibly.
The `filters.py` example in the `examples` directory demonstrates this.

A very useful resource for filters is the `SVO service <http://svo2.cab.inta-csic.es/svo/theory/fps3/>`_, which has transmission curves for the majority of large observatories worldwide::

    from synthesizer.filters import FilterFromSVO, SVOFilterCollection

We can define a single filter using its unique filter code, or as a tuple::

    filt = FilterFromSVO('JWST/NIRCam.F200W')  # use filter code
    filt = FilterFromSVO(('JWST', 'NIRCam', 'F200W'))  # use Tuple

We can also define collections of filters, and plot their transmission curves::

    fs = [f'JWST/NIRCam.{f}' for f in ['F200W', 'F277W']]  # a list of filter codes
    fc = SVOFilterCollection(fs)
    fc.plot_transmission_curves()

.. image:: images/transmission_curves.png

If you just need a simple top hat filter you can use `TopHatFilterCollection` to define this, giving it a name, effective wavelength and full-width half maximum (FWHM)::

    fs = [('U', {'lam_eff': 3650, 'lam_fwhm': 660})]
    fc = TopHatFilterCollection(fs)
    fc.plot_transmission_curves()

.. image:: images/top_hat.png

Finally, we also define the commonly used UVJ filters as a named collection::

    fc = UVJ()
    fc.plot_transmission_curves()

.. image:: images/UVJ.png
                             

InterGalactic Medium
====================

`Synthesizer` provides some common parametrisations of the transmission through the InterGalactic Medium (IGM)::

    from synthesizer.igm import Madau96, Inoue14

We can plot Madau+96 and Inoue+94 at a range of redshifts::

    import matplotlib.pyplot as plt
    import numpy as np

    lam = np.arange(0, 50000)
    z = 7.

    for z in [3., 5., 7.]:
        for IGM, ls in zip([Inoue14, Madau96], ['-', ':']):
            igm = IGM()
            plt.plot(lam, igm.T(z, lam), ls=ls)


    plt.ylim([0, 1.1])
    plt.show()

.. image:: images/igm.png

Abundances
==========

`Synthesizer` contains an abundances object where unique abundance patterns can be defined and explored::

    from synthesizer.abundances_sw import Abundances, plot_abundance_patterns

We can generate a default abundance pattern, given some values for the total metallicity, the abundance of Carbon in CO (*not yet implemented*) the dust to metals ratio (`d2m`), some assumed dust scaling (default: Dopita+2013), and a value for the alpha enhancement::

    Z = 0.01
    CO = 0.0
    d2m = None
    scaling = None
    alpha = 0.0

Create the `AbundancePattern` object, and plot::

    a = abundances.generate_abundances(Z, alpha, CO, d2m, scaling=scaling)

    a.plot_abundances()

.. image:: images/abundances.png


Generate an alpha enhanced abundance pattern while keeping the same total metallicity default abundance pattern::

    alpha = 0.6
    a_ae = abundances.generate_abundances(Z, alpha, CO, d2m, scaling=scaling)

    # [O/Fe], should be 0.4, comes out as 0.4
    # TODO: not clear what correct value should be
    print(a_ae.solar_relative_abundance('O', ref_element='Fe'))

    plot_abundance_patterns([a, a_ae], ['default', r'\alpha = 0.6'],
                                  show=True, ylim=[-7., -3.])


.. image:: images/abundance_comparison.png

