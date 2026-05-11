"""Instrument interfaces and helpers.

This package exposes two user-facing ways to construct instruments:

- `Instrument`, a backwards-compatible convenience constructor that
  dispatches to the most appropriate concrete instrument type.
- Specialised concrete classes such as `PhotometricInstrument`,
  `PhotometricImager`, `SpectroscopicInstrument`, and
  `IntegratedFieldUnit`.

The shared abstract interface for all concrete instruments is
`InstrumentBase`.

`InstrumentCollection` remains the common container for combining one or more
instrument instances regardless of their concrete type.
"""

from synthesizer.instruments.filters import UVJ, Filter, FilterCollection
from synthesizer.instruments.instrument_base import InstrumentBase
from synthesizer.instruments.instrument import Instrument
from synthesizer.instruments.photometric_instrument import PhotometricInstrument
from synthesizer.instruments.photometric_imager import PhotometricImager
from synthesizer.instruments.spectroscopic_instrument import (
    SpectroscopicInstrument,
)
from synthesizer.instruments.integrated_field_unit import IntegratedFieldUnit
from synthesizer.instruments.instrument_collection import InstrumentCollection
from synthesizer.instruments import photometric_noise
from synthesizer.instruments import premade as _premade

# Re-export premade instruments explicitly
AVAILABLE_INSTRUMENTS = _premade.__all__
globals().update({name: getattr(_premade, name) for name in AVAILABLE_INSTRUMENTS})

from synthesizer.instruments.utils import (
    get_lams_from_resolving_power,
    print_premade_instruments,
)

__all__ = [
    "Instrument",
    "InstrumentBase",
    "PhotometricInstrument",
    "PhotometricImager",
    "SpectroscopicInstrument",
    "IntegratedFieldUnit",
    "InstrumentCollection",
    "UVJ",
    "Filter",
    "FilterCollection",
    "get_lams_from_resolving_power",
    "print_premade_instruments",
    *AVAILABLE_INSTRUMENTS,
]
