"""Instrument interfaces and helpers.

This package contains the core instrument abstractions used throughout
Synthesizer. Instruments are used to describe how observational products such
as photometry, images, spectra, and resolved spectroscopy should be generated
or post-processed.

Users can either construct a specialised instrument class directly, or use the
backwards-compatible :class:`Instrument` factory which dispatches to the
appropriate specialised class based on the supplied arguments.

The main specialised instrument classes are:

- :class:`PhotometricInstrument` for integrated photometry,
- :class:`PhotometricImager` for photometric imaging,
- :class:`SpectroscopicInstrument` for one-dimensional spectroscopy, and
- :class:`IntegratedFieldUnit` for spatially resolved spectroscopy.

All specialised instrument classes share the :class:`InstrumentBase`
interface, while :class:`InstrumentCollection` provides the common container
for working with multiple instruments at once.
"""
from synthesizer.instruments.filters import UVJ, Filter, FilterCollection
from synthesizer.instruments.instrument import Instrument
from synthesizer.instruments.instrument_collection import InstrumentCollection
from synthesizer.instruments import photometric_noise
from synthesizer.instruments import premade as _premade

# Re-export premade instruments explicitly so they appear as top-level package
# members alongside the core instrument classes
AVAILABLE_INSTRUMENTS = _premade.__all__
globals().update({name: getattr(_premade, name) for name in AVAILABLE_INSTRUMENTS})

from synthesizer.instruments.utils import (
    get_lams_from_resolving_power,
    print_premade_instruments,
)

__all__ = [
    "Instrument",
    "InstrumentCollection",
    "UVJ",
    "Filter",
    "FilterCollection",
    "get_lams_from_resolving_power",
    "print_premade_instruments",
    *AVAILABLE_INSTRUMENTS,
]
