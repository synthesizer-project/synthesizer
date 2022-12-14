Filters
*******

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
