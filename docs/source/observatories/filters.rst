Filters
=======

Photometric filters in `synthesizer` are defined using two dedicated objects:

- A ``Filter`` object: individual filters defining a filters wavelength coverage and transmission, with related methods and attributes.

- A ``FilterCollection`` object: a collection of ``Filters`` that behaves like a list and a dictionary, with extra attributes and methods to efficiently work with multiple ``Filter`` objects.

We provide a number of different ways to define a ``Filter`` or set of ``Filters``:

- Generic: A generic filter simply requires a user defined wavelength array and transmission curve to initialise. A user can then define any arbitrary filter they like using this functionality. 

- Top Hat: A top hat filter's transmission is 1 in a particular range and 0 everywhere else. These are either defined by a minimum and maximum wavelength of transmission, or by the effective wavelength of the transmission and its full width half maximum (FWHM).

- SVO: We also provide an interface to the `Spanish Virtual Observatory (SVO) filter service <http://svo2.cab.inta-csic.es/theory/fps/>`_.
    The user need only provide the filter code in "Observatory/Instrument.code" format (as shown on the SVO website) to extract the relevant information from this service, and create a ``Filter`` object.

Filters can be used for producing photometry from ``Sed`` objects, as well as for creating monochromatic or RGB images.


.. toctree::
   :maxdepth: 2

   filters_example
