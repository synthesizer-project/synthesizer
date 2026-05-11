Instruments
================= 

Synthesizer provides a small instrument hierarchy for representing different
observing modes. These can correspond to real observatories, such as JWST,
Euclid, or Hubble, or they can be entirely user defined.

The main user-facing entry points are:

- ``Instrument``: a backwards-compatible convenience constructor that
  dispatches to the most appropriate concrete instrument type. ``Instrument``
  is a factory, not the root of the class hierarchy.
- ``InstrumentCollection``: a container for combining one or more instruments.

The specialised concrete instrument classes are:

- ``PhotometricInstrument`` for integrated photometry
- ``PhotometricImager`` for imaging-capable photometric instruments
- ``SpectroscopicInstrument`` for one-dimensional spectroscopy
- ``IntegratedFieldUnit`` for resolved spectroscopy

At the implementation level all concrete instrument classes share the
``InstrumentBase`` interface. Most users should either construct the
specialised classes directly or use ``Instrument(...)`` as a convenience
factory.

As well as allowing the user to define arbitrary instruments, Synthesizer also
provides a suite of premade instruments that can be imported directly or loaded
from cached files when larger datasets such as PSFs and noise arrays are
required.

Typical examples are:

.. code-block:: python

   phot = PhotometricInstrument(label="UVJ", filters=UVJ())

   imager = PhotometricImager(
       label="HST-like",
       filters=filters,
       resolution=0.1 * arcsecond,
   )

   spec = SpectroscopicInstrument(label="Spec", lam=lam)

   ifu = IntegratedFieldUnit(
       label="IFU",
       lam=lam,
       resolution=0.5 * arcsecond,
   )

The equivalent convenience-constructor form is:

.. code-block:: python

   inst = Instrument(label="HST-like", filters=filters,
                     resolution=0.1 * arcsecond)

which dispatches to ``PhotometricImager``.

In this section we detail creating and working with these objects.

.. toctree::
   :maxdepth: 1

   instrument_example 
   premade_instruments 
   filters
