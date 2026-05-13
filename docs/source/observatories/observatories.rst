Instruments
=================

Synthesizer provides a small instrument hierarchy for describing the technical
configuration of a synthetic observation. Instruments can correspond to real
observatories such as JWST, Euclid, or Hubble, or they can be entirely user
defined.

For most workflows you should construct the specialised instrument class that
matches the observing mode you want:

- ``PhotometricInstrument`` for integrated photometry
- ``PhotometricImager`` for photometry plus resolved imaging
- ``SpectroscopicInstrument`` for one-dimensional spectroscopy
- ``IntegratedFieldUnit`` for resolved spectroscopy and spectral cubes
- ``InstrumentCollection`` for combining one or more instruments

The specialised classes are the primary user-facing interface. ``Instrument``
is still available as a convenience factory that dispatches to the appropriate
specialised class, but it is optional.

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

Each class carries the capabilities needed for that observing mode.

- ``PhotometricInstrument`` stores filters and optional photometric depth or
  SNR information
- ``PhotometricImager`` adds spatial resolution together with optional PSFs,
  ``noise_maps``, and ``noise_source_maps``
- ``SpectroscopicInstrument`` stores a wavelength grid for integrated spectra
- ``IntegratedFieldUnit`` combines a wavelength grid and spatial resolution for
  resolved spectroscopy

Synthesizer also provides a suite of premade instruments that can be imported
directly or loaded from cached files when larger datasets such as PSFs and
noise arrays are required.

If you prefer the convenience factory, the equivalent constructor form is:

.. code-block:: python

   inst = Instrument(
       "HST-like",
       filters=filters,
       resolution=0.1 * arcsecond,
   )

which dispatches to ``PhotometricImager``.

In this section we detail creating and working with these instrument types.

.. toctree::
   :maxdepth: 1

   instrument_example 
   premade_instruments 
   filters

