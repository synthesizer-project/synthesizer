Filters
=======

In ``synthesizer`` we encapsulate all filter based functionality in two objects: 

- A ``Filter``: Individual filters with methods and attributes to work with a filters wavelength coverage and transmission.

- A ``FilterCollection``: A collection of ``Filter``'s that behaves like a list with extra attributes and methods to efficiently work with multiple `Filter`s.

We provide a number of different ways to define a ``Filter`` or set of ``Filter``'s:

- Generic: A generic filter simply requires a user defined wavelength array and transmission curve to initialise. As such a user can define any arbitrary filter they like using this functionality. 

- Top Hat: A top hat filter's transmission is 1 in a particular range and 0 everywhere else. These are either defined by a minimum and maximum wavelength of transmission or by the effective wavelength of the transmission and its full width half maximum (FWHM).

- SVO: We also provide an interface to the `Spanish Virtual Observatory (SVO) filter service <http://svo2.cab.inta-csic.es/theory/fps/>`_. The user need only provide the filter code in "Observatory/Instrument.code" format (as shown on SVO) to extract the relevant information from this service and create a ``Filter`` object.


.. toctree::
   :maxdepth: 1

   filters_example
