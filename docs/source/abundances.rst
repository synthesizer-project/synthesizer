Abundances
**********

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
