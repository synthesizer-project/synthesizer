from synthesizer.instruments.filters import UVJ, Filter, FilterCollection
from synthesizer.instruments.instrument import Instrument
from synthesizer.instruments.instrument_collection import InstrumentCollection
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
    "InstrumentCollection",
    "UVJ",
    "Filter",
    "FilterCollection",
    "get_lams_from_resolving_power",
    "print_premade_instruments",
    *AVAILABLE_INSTRUMENTS,
]
